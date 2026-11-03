"""FormCoach AI — сервер для мобильного сайта. Запуск: uvicorn server:app --host 0.0.0.0 --port 8000

Сервер не содержит своей AI-логики: он принимает видео с телефона, вызывает тот же
код, что и Streamlit-приложение (formcoach/: HMM, агент, классификатор), и отдаёт
результат в JSON. Интерфейс лежит в папке web/ (HTML + CSS + JavaScript).
"""

import base64
import json
import tempfile
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from formcoach import synthetic as syn
from formcoach.agent import ERROR_TEXT, SquatRules
from formcoach.classifier import CLASS_RU, DATA as ML_DATA, FormClassifier
from formcoach.exercises import (DEMO_ERRORS, EXERCISES, ExerciseLiveSession, ExerciseSpec, analyze_exercise,
                                 default_hmm, demo_session, exercise_features)
from formcoach.features import squat_features
from formcoach.hmm import STATE_NAMES, squat_hmm
from formcoach.live import LiveSession
from formcoach.pipeline import Analysis, analyze_squat
from formcoach.pose import draw_skeleton, extract_pose, read_frame

ROOT = Path(__file__).resolve().parent
WEB = ROOT / "web"
MAX_UPLOAD_MB = 100
MAX_SECONDS = 90  # длиннее не обрабатываем: на бесплатном сервере это слишком долго
VIDEO_TYPES = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".3gp", ".m4v"}
DEMO_SPECS = {"shallow": syn.SHALLOW, "lean": syn.LEANING, "heels": syn.HEELS_UP, "fast": syn.TOO_FAST}
ERROR_SHORT = {"shallow": "неглубоко", "lean": "наклон", "heels": "пятки", "fast": "быстро"}

# ---------------- два языка: русский и казахский ----------------
# Вся AI-логика (formcoach/) пишет тексты по-русски. Сайт показывает их на выбранном языке,
# поэтому в ответе каждый текст лежит сразу в двух вариантах: {"ru": ..., "kk": ...}.
# Тогда переключение языка на телефоне не требует повторного анализа видео.
LANGS = ("ru", "kk")
TEXT = {
    "ru": {
        "short": ERROR_SHORT,
        "error": ERROR_TEXT,
        "phases": STATE_NAMES,
        "classes": CLASS_RU,
        "live": {"danger": "Опасный наклон! Высокая нагрузка на поясницу", "lean": "Держи спину ровнее",
                 "heels": "Пятки на пол", "walking": "Похоже на ходьбу, не засчитываю"},
        "rep_ok": "Повторение {n}: отлично!",
        "rep_bad": "Повторение {n}: {text}",
    },
    "kk": {
        "short": {"shallow": "таяз", "lean": "еңкею", "heels": "өкше", "fast": "жылдам"},
        "error": {
            "shallow": "Таяз отырыс — төменірек түс (санды еденге параллельге жақындат)",
            "lean": "Дене алға қатты еңкейді — кеудеңді жоғары ұста",
            "heels": "Өкше еденнен көтеріледі — салмақты бүкіл табанға сал",
            "fast": "Тым жылдам — бақылап, баяу түс",
        },
        "phases": ["ЖОҒАРЫ", "ТҮСУ", "ТӨМЕН", "КӨТЕРІЛУ"],
        "classes": {"good": "дұрыс", "bad_back": "арқа", "bad_heel": "өкше"},
        "live": {"danger": "Қауіпті еңкею! Белге түсетін жүктеме жоғары", "lean": "Арқаңды түзу ұста",
                 "heels": "Өкшені еденге қой", "walking": "Жүріске ұқсайды, есептемеймін"},
        "rep_ok": "{n}-қайталау: керемет!",
        "rep_bad": "{n}-қайталау: {text}",
    },
}
# Сообщения об ошибках сервера — на языке, который прислал телефон
ERRORS = {
    "ru": {
        "bad_settings": "Настройки переданы в неверном формате.",
        "bad_param": "Параметр «{key}» должен быть числом от {lo} до {hi}.",
        "not_video": "Это не видеофайл. Подходят mp4, mov, webm, avi, mkv, 3gp.",
        "empty": "Файл пустой.",
        "too_big": "Видео больше {mb} МБ. Сними покороче (20–60 секунд хватает).",
        "too_long": "Видео длиннее {sec} секунд. Обрежь его или сними 5–10 повторений.",
        "no_person": "Человек найден только на {ratio:.0%} кадров. Сними так, чтобы всё тело было в кадре.",
        "failed": "Не получилось обработать видео: {e}",
        "unknown_errors": "Неизвестные ошибки: {names}",
        "unknown_exercise": "Неизвестное упражнение: {name}",
    },
    "kk": {
        "bad_settings": "Баптаулар қате форматта жіберілді.",
        "bad_param": "«{key}» параметрі {lo} мен {hi} аралығындағы сан болуы керек.",
        "not_video": "Бұл видеофайл емес. mp4, mov, webm, avi, mkv, 3gp файлдары жарайды.",
        "empty": "Файл бос.",
        "too_big": "Видео {mb} МБ-тан үлкен. Қысқарақ түсір (20–60 секунд жеткілікті).",
        "too_long": "Видео {sec} секундтан ұзын. Оны қысқарт немесе 5–10 қайталау түсір.",
        "no_person": "Адам кадрлардың тек {ratio:.0%} бөлігінде табылды. Бүкіл дене кадрға сиятындай түсір.",
        "failed": "Видеоны өңдеу мүмкін болмады: {e}",
        "unknown_errors": "Белгісіз қателер: {names}",
        "unknown_exercise": "Белгісіз жаттығу: {name}",
    },
}


def check_lang(lang: str | None) -> str:
    return lang if lang in LANGS else "ru"


def fail(status: int, lang: str, msg: str, **kw) -> HTTPException:
    """Ошибка для телефона: текст msg из ERRORS на нужном языке, kw подставляются в {..}."""
    return HTTPException(status, ERRORS[lang][msg].format(**kw))


def both(make) -> dict:
    """make(тексты_языка) -> строка; возвращает {"ru": ..., "kk": ...}."""
    return {lang: make(TEXT[lang]) for lang in LANGS}


def texts_for(exercise: str) -> dict:
    """Тексты упражнения на двух языках в одном виде: {"ru": {...}, "kk": {...}}.
    Для приседа — таблица TEXT выше, для остальных — тексты из formcoach/exercises.py."""
    if exercise == "squat":
        return TEXT
    spec = EXERCISES[exercise]
    return {lang: {**spec.text[lang], "classes": {}, "rep_ok": TEXT[lang]["rep_ok"], "rep_bad": TEXT[lang]["rep_bad"]}
            for lang in LANGS}


def feedback_text(f, reps_by_number: dict, texts: dict = TEXT) -> dict:
    """Подсказка агента на двух языках: собираем её заново по коду (агент хранит только русский текст)."""
    pick = lambda make: {lang: make(texts[lang]) for lang in LANGS}  # noqa: E731
    if f.code in texts["ru"]["live"]:
        return pick(lambda T: T["live"][f.code])
    if f.code == "rep" and f.rep in reps_by_number:
        r = reps_by_number[f.rep]
        if r.good:
            return pick(lambda T: T["rep_ok"].format(n=r.number))
        return pick(lambda T: T["rep_bad"].format(n=r.number, text="; ".join(T["error"][e] for e in r.errors)))
    return {lang: f.message for lang in LANGS}

app = FastAPI(title="FormCoach AI")


@lru_cache(maxsize=1)
def load_classifier() -> FormClassifier | None:
    return FormClassifier.load()


def parse_settings(raw: str | None, lang: str = "ru") -> tuple[dict, SquatRules]:
    """Настройки из интерфейса (JSON-строка) -> параметры HMM и правила агента.
    Неизвестные ключи игнорируем, значения проверяем по тем же границам, что и ползунки."""
    try:
        s = json.loads(raw) if raw else {}
    except json.JSONDecodeError:
        raise fail(400, lang, "bad_settings")
    if not isinstance(s, dict):
        raise fail(400, lang, "bad_settings")

    def num(key, default, lo, hi):
        v = s.get(key, default)
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not lo <= v <= hi:
            raise fail(400, lang, "bad_param", key=key, lo=lo, hi=hi)
        return float(v)

    hmm = dict(top_angle=num("top_angle", 165, 140, 180), bottom_angle=num("bottom_angle", 85, 50, 130),
               stay_prob=num("stay_prob", 0.9, 0.5, 0.99))
    d = SquatRules()
    rules = SquatRules(
        max_thigh_angle=num("max_thigh_angle", d.max_thigh_angle, -10, 70),
        max_torso_lean=num("max_torso_lean", d.max_torso_lean, 15, 70),
        max_heel_lift=num("max_heel_lift", d.max_heel_lift, 0.05, 0.5),
        min_rep_seconds=num("min_rep_seconds", d.min_rep_seconds, 0.3, 3.0),
    )
    return hmm, rules


def check_exercise(exercise: str | None, lang: str) -> str:
    exercise = exercise or "squat"
    if exercise != "squat" and exercise not in EXERCISES:
        raise fail(400, lang, "unknown_exercise", name=exercise)
    return exercise


def parse_exercise_settings(spec: ExerciseSpec, raw: str | None, lang: str = "ru") -> tuple[dict, dict]:
    """Как parse_settings, но для упражнения из exercises.py: пороги берём из его описания (spec.fields)."""
    try:
        s = json.loads(raw) if raw else {}
    except json.JSONDecodeError:
        raise fail(400, lang, "bad_settings")
    if not isinstance(s, dict):
        raise fail(400, lang, "bad_settings")

    def num(key, default, lo, hi):
        v = s.get(key, default)
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not lo <= v <= hi:
            raise fail(400, lang, "bad_param", key=key, lo=lo, hi=hi)
        return float(v)

    hmm = default_hmm(spec)
    hmm.update(top_angle=num("top_angle", spec.top, 140, 180), bottom_angle=num("bottom_angle", spec.bottom, 50, 130),
               stay_prob=num("stay_prob", 0.9, 0.5, 0.99))
    if hmm["bottom_angle"] >= hmm["top_angle"]:
        raise fail(400, lang, "bad_param", key="bottom_angle", lo=50, hi=hmm["top_angle"])
    rules = {f.key: num(f.key, f.default, f.lo, f.hi) for f in spec.fields}
    return hmm, rules


def settings_for(exercise: str, raw: str | None, lang: str):
    return parse_settings(raw, lang) if exercise == "squat" else parse_exercise_settings(EXERCISES[exercise], raw, lang)


def crop_to_body(img: np.ndarray, points: np.ndarray, aspect: float = 4 / 3, margin: float = 0.15) -> np.ndarray:
    """Вырезаем из кадра прямоугольник 4:3 вокруг человека — на вертикальном видео иначе в карточке виден потолок."""
    xy = points[:, :2][~np.isnan(points[:, 0])]
    if len(xy) == 0:
        return img
    h, w = img.shape[:2]
    (x0, y0), (x1, y1) = xy.min(axis=0), xy.max(axis=0)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    bh = (y1 - y0) * (1 + 2 * margin)
    bw = (x1 - x0) * (1 + 2 * margin)
    bw, bh = max(bw, bh * aspect), max(bh, bw / aspect)  # дотягиваем до пропорции 4:3
    bw, bh = min(bw, w), min(bh, h)
    left = int(np.clip(cx - bw / 2, 0, w - bw))
    top = int(np.clip(cy - bh / 2, 0, h - bh))
    return img[top:top + int(bh), left:left + int(bw)]


def thumbnail(video_path: Path | None, landmarks: np.ndarray, frame: int, width: int = 360) -> str | None:
    """Кадр нижней точки со скелетом -> маленькая JPEG-картинка в base64 (для <img src=...>).
    video_path=None (живая камера: видео не присылали) — рисуем только скелет на тёмном фоне."""
    if video_path is None:
        pts = landmarks[frame]
        if np.isnan(pts[0, 0]):
            return None
        h, w = int(np.nanmax(pts[:, 1])) + 40, int(np.nanmax(pts[:, 0])) + 40
        rgb = np.full((max(h, 10), max(w, 10), 3), (22, 25, 30), np.uint8)
    else:
        rgb = read_frame(video_path, frame)
    if rgb is None:
        return None
    img = draw_skeleton(rgb, landmarks[frame], color=(198, 255, 61))
    img = crop_to_body(img, landmarks[frame])
    h, w = img.shape[:2]
    if w > width:
        img = cv2.resize(img, (width, int(h * width / w)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 80])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode() if ok else None


def clean(x) -> float | None:
    """NaN нельзя положить в JSON — превращаем его в null."""
    return None if x is None or np.isnan(x) else round(float(x), 2)


def rep_json(r, v, exercise: str, texts: dict) -> dict:
    """Одно повторение для карточки. У приседа свои поля (min_knee, ...), у других упражнений —
    общий список metrics: подпись, значение, единица и превышен ли порог."""
    pick = lambda make: {lang: make(texts[lang]) for lang in LANGS}  # noqa: E731
    out = {
        "number": r.number,
        "good": r.good,
        "errors": [{"key": e, "short": pick(lambda T: T["short"][e]), "text": pick(lambda T: T["error"][e])} for e in r.errors],
        "seconds": clean(r.seconds),
        "ml": None if v is None else {
            "label": v.label, "label_text": pick(lambda T: T["classes"][v.label]), "no_feet": v.no_feet,
            "probs": {k: clean(p) for k, p in v.probs.items()},
        },
    }
    if exercise == "squat":
        out.update(min_knee=clean(r.min_knee), min_thigh=clean(r.min_thigh), max_lean=clean(r.max_lean),
                   max_heel=clean(r.max_heel))
        return out
    spec = EXERCISES[exercise]
    over = {(c.feature, c.agg) for c in spec.checks if c.code in r.errors}
    out["metrics"] = [{"label": pick(lambda T: T["metric"][name]), "value": clean(r.values.get(f"{name}:{agg}", np.nan)),
                       "unit": unit, "over": (name, agg) in over} for name, agg, unit in spec.metrics]
    return out


def to_json(a: Analysis, hmm_params: dict, video_path: Path | None = None, landmarks: np.ndarray | None = None,
            exercise: str = "squat") -> dict:
    """Результат анализа в виде, удобном для интерфейса."""
    texts = texts_for(exercise)
    pick = lambda make: {lang: make(texts[lang]) for lang in LANGS}  # noqa: E731
    main = "knee_angle" if exercise == "squat" else EXERCISES[exercise].main
    bottom_value = (lambda r: r.min_knee) if exercise == "squat" else (lambda r: r.values[f"{main}:min"])
    reps = []
    for r, v in zip(a.reps, a.ml):
        item = rep_json(r, v, exercise, texts)
        item.update(start_s=round(r.start / a.fps, 2), end_s=round(r.end / a.fps, 2),
                    image=thumbnail(video_path, landmarks, r.bottom_frame) if landmarks is not None else None)
        reps.append(item)
    # для графика достаточно ~300 точек, иначе ответ получается тяжёлым
    step = max(1, len(a.phases) // 300)
    good = sum(r.good for r in a.reps)
    by_number = {r.number: r for r in a.reps}
    return {
        "exercise": exercise,
        "summary": {"reps": len(a.reps), "good": good, "bad": len(a.reps) - good,
                    "seconds": round(len(a.phases) / a.fps, 1)},
        "reps": reps,
        "feedback": [{"t": round(f.frame / a.fps, 1), "level": f.level, "message": feedback_text(f, by_number, texts)}
                     for f in a.feedback],
        "series": {
            "t": [round(i / a.fps, 2) for i in range(0, len(a.phases), step)],
            "knee": [clean(x) for x in a.features[main][::step]],  # главный угол (у приседа — колено)
            "phase": [int(p) for p in a.phases[::step]],
            "bottoms": [{"t": round(r.bottom_frame / a.fps, 2), "knee": clean(bottom_value(r)), "n": r.number} for r in a.reps],
        },
        "phase_names": pick(lambda T: T["phases"]),
        "hmm": {"trans": np.round(squat_hmm(**hmm_params).trans, 3).tolist(),
                "log_likelihood": round(a.log_likelihood, 1),
                "per_frame": round(a.log_likelihood / len(a.phases), 2)},
        "has_ml": any(v is not None for v in a.ml),
    }


@app.post("/api/analyze")
async def analyze_video(video: UploadFile = File(...), settings: str | None = Form(None), lang: str = Form("ru"),
                        exercise: str = Form("squat")):
    lang = check_lang(lang)
    exercise = check_exercise(exercise, lang)
    hmm_params, rules = settings_for(exercise, settings, lang)
    suffix = Path(video.filename or "").suffix.lower()
    if suffix not in VIDEO_TYPES:
        raise fail(400, lang, "not_video")
    data = await video.read()
    if not data:
        raise fail(400, lang, "empty")
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise fail(413, lang, "too_big", mb=MAX_UPLOAD_MB)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / f"upload{suffix}"
        path.write_bytes(data)
        cap = cv2.VideoCapture(str(path))
        fps, n = cap.get(cv2.CAP_PROP_FPS) or 30.0, cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.release()
        if n > 0 and n / fps > MAX_SECONDS:
            raise fail(400, lang, "too_long", sec=MAX_SECONDS)
        try:
            seq = extract_pose(path)
            if seq.detected_ratio < 0.5:
                raise fail(422, lang, "no_person", ratio=seq.detected_ratio)
            if exercise == "squat":
                a = analyze_squat(squat_features(seq), seq.fps, hmm_params, rules,
                                  landmarks=seq.landmarks, classifier=load_classifier())
            else:
                spec = EXERCISES[exercise]
                a = analyze_exercise(spec, exercise_features(spec, seq), seq.fps, hmm_params, rules)
        except (ValueError, FileNotFoundError) as e:
            raise fail(422, lang, "failed", e=e)
        return to_json(a, hmm_params, path, seq.landmarks, exercise)


@app.get("/api/demo")
def demo(errors: str = "shallow,lean", settings: str | None = None, lang: str = "ru", exercise: str = "squat"):
    """Демо без видео: синтетический спортсмен делает выбранные ошибки (между ними — правильные повторения)."""
    lang = check_lang(lang)
    exercise = check_exercise(exercise, lang)
    hmm_params, rules = settings_for(exercise, settings, lang)
    chosen = [e for e in errors.split(",") if e]
    allowed = DEMO_SPECS if exercise == "squat" else DEMO_ERRORS[exercise]
    unknown = [e for e in chosen if e not in allowed]
    if unknown:
        raise fail(400, lang, "unknown_errors", names=", ".join(unknown))
    fps = 30.0
    if exercise != "squat":
        spec = EXERCISES[exercise]
        a = analyze_exercise(spec, demo_session(spec, chosen, fps), fps, hmm_params, rules)
        return to_json(a, hmm_params, exercise=exercise)
    plan = [syn.GOOD]
    for e in chosen:
        plan += [DEMO_SPECS[e], syn.GOOD]
    return to_json(analyze_squat(syn.synthetic_session(plan, fps=fps), fps, hmm_params, rules), hmm_params)


def exercise_info(spec: ExerciseSpec) -> dict:
    """Что нужно сайту, чтобы построить настройки и демо упражнения: ползунки, значения по умолчанию, ошибки."""
    return {
        "name": {lang: spec.text[lang]["name"] for lang in LANGS},
        "defaults": {"top_angle": spec.top, "bottom_angle": spec.bottom, "stay_prob": 0.9, **spec.default_rules()},
        "fields": [{"key": f.key, "min": f.lo, "max": f.hi, "step": f.step, "unit": f.unit,
                    "label": {lang: spec.text[lang]["fields"][f.key][0] for lang in LANGS},
                    "help": {lang: spec.text[lang]["fields"][f.key][1] for lang in LANGS}} for f in spec.fields],
        "errors": {e: {lang: spec.text[lang]["short"][e] for lang in LANGS} for e in DEMO_ERRORS[spec.key]},
        "phases": {lang: spec.text[lang]["phases"] for lang in LANGS},
    }


@app.get("/api/info")
def info():
    """Настройки по умолчанию и метрики моделей — для экрана «Как это работает»."""
    d = SquatRules()
    out = {
        "defaults": {"top_angle": 165, "bottom_angle": 85, "stay_prob": 0.9, "max_thigh_angle": d.max_thigh_angle,
                     "max_torso_lean": d.max_torso_lean, "max_heel_lift": d.max_heel_lift,
                     "min_rep_seconds": d.min_rep_seconds},
        "errors": ERROR_SHORT,
        "models": None, "retrain": None,
        "exercises": {key: exercise_info(spec) for key, spec in EXERCISES.items()},
    }
    clf_report, retrain_report = ML_DATA / "classifier_report.json", ML_DATA / "retrain_report.json"
    if clf_report.exists():
        rep = json.loads(clf_report.read_text())
        out["models"] = [{"name": k, "cv_f1": round(v["cv"]["f1_macro"], 3), "test_f1": round(v["test"]["f1_macro"], 3)}
                         for k, v in rep["results"].items()]
    if retrain_report.exists():
        rr = json.loads(retrain_report.read_text())
        out["retrain"] = {"videos": rr["videos"], "reps": rr["reps"]}
    return out


# ---------------- живая камера ----------------
# Телефон сам находит позу (MediaPipe в браузере) и присылает по WebSocket только 33 точки тела на кадр.
# Сервер прогоняет их через LiveSession (признаки -> HMM-фильтрация -> агент) и сразу отвечает подсказками.
# Протокол (JSON):
#   телефон -> {"type": "start", "settings": {...}, "lang": "kk"}      сервер -> {"type": "ready"}
#   телефон -> {"type": "frame", "t": секунды, "lm": [132 числа] | null} сервер -> {"type": "state", ...}
#   телефон -> {"type": "stop"}                                         сервер -> {"type": "result", "data": {...}}
MAX_LIVE_FRAMES = 30 * 60 * 10  # не больше 10 минут при 30 кадр/с — защита памяти сервера
LIVE_ERRORS = {
    "ru": {"protocol": "Ошибка связи с камерой. Обнови страницу.", "short": "Слишком короткий подход — сделай хотя бы одно повторение.",
           "long": "Подход слишком длинный (больше 10 минут). Начни новый."},
    "kk": {"protocol": "Камерамен байланыс қатесі. Бетті жаңарт.", "short": "Тым қысқа — кем дегенде бір қайталау жаса.",
           "long": "Тым ұзақ (10 минуттан көп). Жаңасын баста."},
}


def live_event(f, session, exercise: str = "squat") -> dict:
    """Подсказка агента для телефона: код (по нему играется голосовая фраза) + текст на двух языках."""
    reps = {r.number: r for r in session.agent.reps}
    event = {"code": f.code, "level": f.level, "rep": f.rep, "message": feedback_text(f, reps, texts_for(exercise))}
    if f.code == "rep" and f.rep in reps:
        event["good"] = reps[f.rep].good
        event["errors"] = reps[f.rep].errors
    return event


def parse_frame(msg: dict):
    """Проверяем кадр от телефона: время — число, точки — 33 x 4 конечных числа или null."""
    t = msg.get("t")
    lm = msg.get("lm")
    if isinstance(t, bool) or not isinstance(t, (int, float)) or not np.isfinite(t):
        raise ValueError("bad time")
    if lm is None:
        return float(t), None
    pts = np.asarray(lm, dtype=float)
    if pts.size != 33 * 4 or not np.all(np.isfinite(pts)):
        raise ValueError("bad landmarks")
    return float(t), pts.reshape(33, 4)


@app.websocket("/ws/live")
async def live(ws: WebSocket):
    await ws.accept()
    lang, session, hmm_params, exercise = "ru", None, None, "squat"
    try:
        while True:
            try:
                msg = await ws.receive_json()
            except ValueError:  # прислали не JSON
                await ws.send_json({"type": "error", "detail": LIVE_ERRORS[lang]["protocol"]})
                continue
            kind = msg.get("type") if isinstance(msg, dict) else None
            try:
                if kind == "start":
                    lang = check_lang(msg.get("lang"))
                    ex = msg.get("exercise") or "squat"
                    exercise = check_exercise(ex if isinstance(ex, str) else "?", lang)
                    hmm_params, rules = settings_for(exercise, json.dumps(msg.get("settings") or {}), lang)
                    if exercise == "squat":
                        session = LiveSession(hmm_params, rules)
                    else:
                        session = ExerciseLiveSession(EXERCISES[exercise], hmm_params, rules)
                    await ws.send_json({"type": "ready", "exercise": exercise,
                                        "phase_names": {lang_: texts_for(exercise)[lang_]["phases"] for lang_ in LANGS}})
                elif kind == "frame" and session is not None:
                    if len(session.times) >= MAX_LIVE_FRAMES:
                        await ws.send_json({"type": "error", "detail": LIVE_ERRORS[lang]["long"]})
                        continue
                    t, pts = parse_frame(msg)
                    out = session.push(t, pts)
                    out["events"] = [live_event(f, session, exercise) for f in out["events"]]
                    await ws.send_json({"type": "state", **out})
                elif kind == "stop" and session is not None:
                    try:
                        a = session.summary(load_classifier())
                    except ValueError:
                        await ws.send_json({"type": "error", "detail": LIVE_ERRORS[lang]["short"]})
                    else:
                        await ws.send_json({"type": "result", "data": to_json(a, hmm_params, None, np.array(session.landmarks), exercise)})
                    session = None
                else:
                    raise ValueError("unknown message")
            except HTTPException as e:  # неверные настройки
                await ws.send_json({"type": "error", "detail": e.detail})
            except (ValueError, TypeError):
                await ws.send_json({"type": "error", "detail": LIVE_ERRORS[lang]["protocol"]})
    except WebSocketDisconnect:
        pass  # телефон закрыл страницу — ничего не храним


# Service worker должен отдаваться из корня сайта, иначе браузер не даст ему управлять всеми страницами
@app.get("/sw.js")
def service_worker():
    return FileResponse(WEB / "sw.js", media_type="application/javascript")


# Всё остальное — файлы интерфейса (index.html, стили, скрипт, иконки)
app.mount("/", StaticFiles(directory=WEB, html=True), name="web")
