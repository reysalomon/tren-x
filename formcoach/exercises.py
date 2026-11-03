"""Другие упражнения (сейчас — становая тяга) на тех же AI-темах, что и приседания.

Цепочка та же, меняется только "что измерять":
  точки тела -> признаки упражнения -> HMM по ГЛАВНОМУ углу (неделя 7: ВЕРХ/ОПУСКАНИЕ/НИЗ/ПОДЪЁМ)
             -> агент-тренер с правилами этого упражнения (неделя 1)

Упражнение описывается данными (ExerciseSpec): главный угол, параметры HMM, список проверок
"признак за повторение больше/меньше порога -> ошибка" и тексты на двух языках. Новое упражнение —
это новое описание + функция признаков одного кадра; HMM, агент и живой режим общие.

Приседания остаются в agent.py/live.py: у них есть классификатор (неделя 8) и особые правила (ходьба).
"""

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from . import pose as P
from .agent import Feedback
from .features import FOOT_MIN_VISIBILITY, joint_angle, smooth
from .hmm import TOP, squat_hmm
from .live import MIN_VISIBILITY, hmm_for_step, next_phase
from .pipeline import Analysis

SIDES = {
    "L": dict(sh=P.L_SHOULDER, el=P.L_ELBOW, wr=P.L_WRIST, hip=P.L_HIP, knee=P.L_KNEE, ankle=P.L_ANKLE, heel=P.L_HEEL, foot=P.L_FOOT),
    "R": dict(sh=P.R_SHOULDER, el=P.R_ELBOW, wr=P.R_WRIST, hip=P.R_HIP, knee=P.R_KNEE, ankle=P.R_ANKLE, heel=P.R_HEEL, foot=P.R_FOOT),
}


@dataclass(frozen=True)
class Check:
    """Правило тренера: если agg(признак) за повторение op порог rules[rule] — ошибка code.
    Пример: Check("squat_it", "knee_angle", "min", "<", "min_knee_angle") — колено согнулось сильнее порога."""

    code: str
    feature: str
    agg: str  # "max" или "min" за повторение
    op: str  # ">" или "<"
    rule: str
    live: bool = False  # подсказывать сразу во время движения, а не только в конце повторения
    at_top: bool = False  # ошибка случается, когда человек УЖЕ встал (прогиб назад) — проверяем и после повторения

    def broken(self, value: float, threshold: float) -> bool:
        if np.isnan(value):  # точку не было видно — не ругаем
            return False
        return bool(value > threshold if self.op == ">" else value < threshold)


@dataclass(frozen=True)
class Field:
    """Порог, который можно менять ползунком в настройках сайта."""

    key: str
    default: float
    lo: float
    hi: float
    step: float
    unit: str = ""


@dataclass(frozen=True)
class ExerciseSpec:
    key: str
    main: str  # главный угол: по нему HMM находит фазы и считаются повторения
    top: float  # главный угол в положении стоя (параметр HMM по умолчанию)
    bottom: float  # главный угол в нижней точке
    speed: float  # типичная скорость изменения главного угла, град/с
    min_dip: float  # главный угол должен уменьшиться хотя бы на столько, иначе это не повторение
    checks: tuple[Check, ...]
    fields: tuple[Field, ...]  # пороги правил (+ min_rep_seconds)
    values: Callable  # (точки кадра (33,4), сторона "L"/"R", направление взгляда) -> признаки кадра
    facing_vote: Callable  # (точки, сторона) -> голос за направление: >0 смотрит вправо, <0 влево
    required: tuple[str, ...]  # точки, которые должны быть видны (имена из SIDES)
    metrics: tuple[tuple[str, str, str], ...]  # (признак, agg, единица) — что показать в карточке повторения
    text: dict  # тексты на двух языках

    def default_rules(self) -> dict[str, float]:
        return {f.key: f.default for f in self.fields}


# ---------------- становая тяга ----------------


def deadlift_values(pts: np.ndarray, side: str, facing: float) -> dict[str, float]:
    """Признаки становой тяги на одном кадре (камера сбоку).

    hip_angle  — угол в тазу (плечо-таз-колено): ~170 стоя, ~80-100 в наклоне. Это главный угол.
    knee_angle — угол колена: слишком маленький — тяга превратилась в присед, почти 180 — ноги прямые.
    lean_back  — наклон корпуса НАЗАД от вертикали, градусы (0 — стоит ровно или наклонён вперёд).
    bar_dist   — насколько кисти (штанга) ушли вперёд/назад от середины стопы, в длинах голени.
    """
    s = SIDES[side]
    xy = pts[:, :2]
    shin = max(float(np.linalg.norm(xy[s["knee"]] - xy[s["ankle"]])), 1e-6)
    # Наклон корпуса со знаком: >0 — вперёд (туда, куда смотрит человек). y в картинке направлен вниз.
    dx = (xy[s["sh"], 0] - xy[s["hip"], 0]) * (1 if facing >= 0 else -1)
    dy = xy[s["hip"], 1] - xy[s["sh"], 1]
    lean_forward = float(np.degrees(np.arctan2(dx, dy)))
    # Середина стопы: между лодыжкой и носком; если носок не видно — просто лодыжка.
    mid = xy[s["ankle"], 0]
    if pts[s["foot"], 3] >= FOOT_MIN_VISIBILITY:
        mid = (xy[s["ankle"], 0] + xy[s["foot"], 0]) / 2
    bar = float(abs(xy[s["wr"], 0] - mid) / shin) if pts[s["wr"], 3] >= MIN_VISIBILITY else np.nan
    return {
        "hip_angle": float(joint_angle(xy[s["sh"]], xy[s["hip"]], xy[s["knee"]])),
        "knee_angle": float(joint_angle(xy[s["hip"]], xy[s["knee"]], xy[s["ankle"]])),
        "lean_back": max(-lean_forward, 0.0),  # наклон вперёд для этой проверки не важен — считаем 0
        "bar_dist": bar,
    }


def deadlift_facing(pts: np.ndarray, side: str) -> float:
    """Куда смотрит человек: носок впереди пятки, а в наклоне плечи уходят вперёд от таза."""
    s = SIDES[side]
    xy = pts[:, :2]
    vote = 0.0
    if min(pts[s["heel"], 3], pts[s["foot"], 3]) >= FOOT_MIN_VISIBILITY:
        vote += float(np.sign(xy[s["foot"], 0] - xy[s["heel"], 0]))
    if joint_angle(xy[s["sh"]], xy[s["hip"]], xy[s["knee"]]) < 140:  # наклонился — плечи точно впереди
        vote += 2 * float(np.sign(xy[s["sh"], 0] - xy[s["hip"], 0]))
    return vote


DEADLIFT = ExerciseSpec(
    key="deadlift",
    main="hip_angle",
    top=170.0,
    bottom=95.0,
    speed=90.0,
    min_dip=40.0,
    checks=(
        Check("bar_far", "bar_dist", "max", ">", "max_bar_dist", live=True),
        Check("lean_back", "lean_back", "max", ">", "max_lean_back", live=True, at_top=True),
        Check("squat_it", "knee_angle", "min", "<", "min_knee_angle"),
        Check("stiff_legs", "knee_angle", "min", ">", "max_knee_angle"),
    ),
    fields=(
        Field("max_bar_dist", 0.5, 0.2, 1.2, 0.05),
        Field("max_lean_back", 15.0, 5.0, 40.0, 1.0, "°"),
        Field("min_knee_angle", 100.0, 70.0, 140.0, 1.0, "°"),
        Field("max_knee_angle", 165.0, 140.0, 180.0, 1.0, "°"),
        Field("min_rep_seconds", 1.0, 0.3, 3.0, 0.1, " s"),
    ),
    values=deadlift_values,
    facing_vote=deadlift_facing,
    required=("sh", "hip", "knee"),
    metrics=(("hip_angle", "min", "°"), ("knee_angle", "min", "°"), ("lean_back", "max", "°"), ("bar_dist", "max", "")),
    text={
        "ru": {
            "name": "Становая тяга",
            "phases": ["ВЕРХ", "ОПУСКАНИЕ", "НИЗ", "ПОДЪЁМ"],
            "error": {
                "bar_far": "Штанга далеко от ног — веди её вплотную по бёдрам и голеням",
                "lean_back": "Заваливаешься назад наверху — просто встань ровно, без прогиба",
                "squat_it": "Колени сгибаются слишком сильно — это уже присед, отводи таз назад",
                "stiff_legs": "Ноги почти прямые — слегка согни колени, береги поясницу",
                "fast": "Слишком быстро — опускай штангу подконтрольно",
            },
            "short": {"bar_far": "штанга далеко", "lean_back": "прогиб назад", "squat_it": "присед",
                      "stiff_legs": "прямые ноги", "fast": "быстро"},
            "live": {"bar_far": "Штангу ближе к ногам", "lean_back": "Не заваливайся назад"},
            "metric": {"hip_angle": "Таз", "knee_angle": "Колено", "lean_back": "Назад", "bar_dist": "Штанга"},
            "fields": {
                "max_bar_dist": ("Штанга от стопы, макс.", "В длинах голени: насколько кисти могут уйти вперёд от середины стопы."),
                "max_lean_back": ("Наклон назад наверху, макс.", ""),
                "min_knee_angle": ("Угол колена внизу, мин.", "Меньше — тяга превращается в присед."),
                "max_knee_angle": ("Угол колена внизу, макс.", "Больше — ноги почти прямые."),
                "min_rep_seconds": ("Мин. время повторения", "Быстрее — «бросает» штангу без контроля."),
            },
        },
        "kk": {
            "name": "Өлі тартылым",
            "phases": ["ЖОҒАРЫ", "ТҮСІРУ", "ТӨМЕН", "КӨТЕРУ"],
            "error": {
                "bar_far": "Штанга аяқтан алыс — оны сан мен сирақты бойлай жақын жүргіз",
                "lean_back": "Жоғарыда артқа шалқаясың — майыспай, тік тұр",
                "squat_it": "Тізе тым қатты бүгіледі — бұл отырып-тұру, жамбасты артқа шығар",
                "stiff_legs": "Аяқ тым түзу — тізені сәл бүк, беліңді сақта",
                "fast": "Тым жылдам — штанганы бақылап түсір",
            },
            "short": {"bar_far": "штанга алыс", "lean_back": "артқа майысу", "squat_it": "отырып-тұру",
                      "stiff_legs": "түзу аяқ", "fast": "жылдам"},
            "live": {"bar_far": "Штанганы аяққа жақында", "lean_back": "Артқа шалқайма"},
            "metric": {"hip_angle": "Жамбас", "knee_angle": "Тізе", "lean_back": "Артқа", "bar_dist": "Штанга"},
            "fields": {
                "max_bar_dist": ("Штанга табаннан, ең көп", "Сирақ ұзындығымен: қол табан ортасынан қаншалықты алға кете алады."),
                "max_lean_back": ("Жоғарыда артқа еңкею, ең көп", ""),
                "min_knee_angle": ("Төменде тізе бұрышы, ең аз", "Аз болса — тартылым отырып-тұруға айналады."),
                "max_knee_angle": ("Төменде тізе бұрышы, ең көп", "Көп болса — аяқ тым түзу."),
                "min_rep_seconds": ("Қайталаудың ең аз уақыты", "Жылдамырақ — штанганы бақылаусыз «тастайды»."),
            },
        },
    },
)

EXERCISES: dict[str, ExerciseSpec] = {"deadlift": DEADLIFT}


def get_exercise(key: str) -> ExerciseSpec:
    if key not in EXERCISES:
        raise ValueError(f"Неизвестное упражнение: {key}")
    return EXERCISES[key]


def default_hmm(spec: ExerciseSpec) -> dict:
    """Параметры HMM по умолчанию. Модель та же, что у приседа (hmm.squat_hmm): 4 фазы по кругу,
    наблюдения — главный угол и его скорость; меняются только углы "стоя" и "внизу"."""
    return {"top_angle": spec.top, "bottom_angle": spec.bottom, "stay_prob": 0.9, "speed": spec.speed}


# ---------------- агент-тренер (неделя 1), общий для упражнений ----------------


@dataclass
class ExerciseRep:
    number: int
    start: int
    end: int
    bottom_frame: int
    seconds: float
    values: dict[str, float]  # "признак:min" / "признак:max" за повторение
    errors: list[str] = field(default_factory=list)

    @property
    def good(self) -> bool:
        return not self.errors


class ExerciseCoachAgent:
    """Тот же рефлекторный агент с моделью мира, что SquatCoachAgent, но правила берутся из ExerciseSpec.

    Восприятие: фаза от HMM + признаки текущего кадра. Модель мира: идёт ли повторение,
    минимумы/максимумы признаков за него, положение стоя перед ним. Действия: подсказки во время
    движения (Check.live) и итог повторения.
    """

    def __init__(self, spec: ExerciseSpec, rules: dict | None = None, fps: float = 30.0):
        self.spec = spec
        self.rules = {**spec.default_rules(), **(rules or {})}
        self.fps = fps
        self.reps: list[ExerciseRep] = []
        self._in_rep = False
        self._standing = -np.inf  # самый большой главный угол стоя перед повторением
        self._warned: set[str] = set()
        self._last_end = -10**9  # кадр конца последнего повторения
        self._tracked = sorted({spec.main} | {c.feature for c in spec.checks} | {m[0] for m in spec.metrics})

    def _start_rep(self, frame: int, v: dict):
        self._in_rep, self._start, self._warned, self._agg = True, frame, set(), {}
        self._bottom_frame, self._min_main = frame, v[self.spec.main]
        self._update(frame, v)

    def _update(self, frame: int, v: dict):
        if v[self.spec.main] < self._min_main:
            self._min_main, self._bottom_frame = v[self.spec.main], frame
        for name in self._tracked:
            x = v.get(name, np.nan)
            self._agg[f"{name}:min"] = np.fmin(self._agg.get(f"{name}:min", np.nan), x)  # fmin/fmax пропускают NaN
            self._agg[f"{name}:max"] = np.fmax(self._agg.get(f"{name}:max", np.nan), x)

    TOP_WINDOW_SEC = 1.0  # сколько ещё следим за "верхними" ошибками после конца повторения

    def _check_after_rep(self, frame: int, v: dict) -> list[Feedback]:
        """Повторение засчитано в момент, когда HMM увидел ВЕРХ, но заваливаются назад обычно чуть позже,
        уже выпрямившись. Поэтому ~1 с после конца повторения ещё проверяем правила с at_top
        и дописываем ошибку к этому же повторению."""
        if not self.reps or (frame - self._last_end) / self.fps > self.TOP_WINDOW_SEC:
            return []
        rep, out = self.reps[-1], []
        for c in self.spec.checks:
            if c.at_top and c.code not in rep.errors and c.broken(v.get(c.feature, np.nan), self.rules[c.rule]):
                rep.errors.insert(len([e for e in rep.errors if e != "fast"]), c.code)
                key = f"{c.feature}:{c.agg}"
                rep.values[key] = float(np.fmax(rep.values.get(key, np.nan), v[c.feature]))
                out.append(Feedback(frame, "warn", self.spec.text["ru"]["live"][c.code], rep.number, c.code))
        return out

    def step(self, frame: int, phase: int, v: dict) -> list[Feedback]:
        actions: list[Feedback] = self._check_after_rep(frame, v)
        if not self._in_rep:
            if phase == TOP:
                self._standing = max(self._standing, v[self.spec.main])
            else:
                self._start_rep(frame, v)
            return actions

        self._update(frame, v)
        for c in self.spec.checks:
            if c.live and c.code not in self._warned and c.broken(v.get(c.feature, np.nan), self.rules[c.rule]):
                self._warned.add(c.code)
                actions.append(Feedback(frame, "warn", self.spec.text["ru"]["live"][c.code], len(self.reps) + 1, c.code))

        if phase == TOP:  # вернулся в положение стоя — повторение закончилось
            self._in_rep = False
            long_enough = (frame - self._start) / self.fps >= 0.4
            deep_enough = self._standing - self._min_main >= self.spec.min_dip
            if long_enough and deep_enough:  # иначе это дрожание, а не повторение
                actions.append(self._finish(frame))
                self._last_end = frame
            self._standing = v[self.spec.main]
        return actions

    def _finish(self, end: int) -> Feedback:
        values = {k: float(x) for k, x in self._agg.items()}
        rep = ExerciseRep(len(self.reps) + 1, self._start, end, self._bottom_frame, (end - self._start) / self.fps, values)
        rep.errors = [c.code for c in self.spec.checks
                      if c.broken(values.get(f"{c.feature}:{c.agg}", np.nan), self.rules[c.rule])]
        if rep.seconds < self.rules["min_rep_seconds"]:
            rep.errors.append("fast")
        self.reps.append(rep)
        if rep.good:
            return Feedback(end, "ok", f"Повторение {rep.number}: отлично!", rep.number, "rep")
        text = "; ".join(self.spec.text["ru"]["error"][e] for e in rep.errors)
        return Feedback(end, "warn", f"Повторение {rep.number}: {text}", rep.number, "rep")


# ---------------- анализ целого видео ----------------


def _required_idx(spec: ExerciseSpec, side: str) -> list[int]:
    return [SIDES[side][j] for j in spec.required]


def _visible(pts: np.ndarray, spec: ExerciseSpec, side: str) -> bool:
    return not np.isnan(pts[:, :2]).any() and pts[_required_idx(spec, side), 3].min() >= MIN_VISIBILITY


def exercise_features(spec: ExerciseSpec, seq: P.PoseSequence, smooth_window: int = 5) -> dict[str, np.ndarray]:
    """Покадровые признаки упражнения для целого видео (как squat_features для приседа)."""
    lm = seq.landmarks
    vis = {side: np.nanmean(lm[:, _required_idx(spec, side), 3]) for side in SIDES}
    if all(np.isnan(x) for x in vis.values()):
        raise ValueError("Человек не найден ни на одном кадре.")
    side = max(vis, key=lambda s: np.nan_to_num(vis[s]))  # сторона, которую камера видит лучше
    seen = [_visible(p, spec, side) for p in lm]
    if not any(seen):
        raise ValueError("Человек не найден ни на одном кадре.")
    facing = sum(spec.facing_vote(p, side) for p, ok in zip(lm, seen) if ok)
    rows = [spec.values(p, side, facing) if ok else None for p, ok in zip(lm, seen)]
    names = list(next(r for r in rows if r is not None))
    feats = {n: smooth(np.array([np.nan if r is None else r[n] for r in rows]), smooth_window) for n in names}
    feats["main_vel"] = np.gradient(feats[spec.main]) * seq.fps
    return feats


def analyze_exercise(spec: ExerciseSpec, features: dict, fps: float, hmm_params: dict | None = None,
                     rules: dict | None = None) -> Analysis:
    """Признаки -> HMM (Витерби по главному углу) -> агент. Классификатора (неделя 8) для этих упражнений нет."""
    main = features[spec.main]
    if len(main) < 2:
        raise ValueError("Слишком короткое видео для анализа.")
    hmm = squat_hmm(**{**default_hmm(spec), **(hmm_params or {})})
    X = np.column_stack([main, features["main_vel"]])
    phases, _ = hmm.viterbi(X)
    agent = ExerciseCoachAgent(spec, rules, fps)
    feedback = []
    for t in range(len(phases)):
        feedback += agent.step(t, int(phases[t]), {k: float(v[t]) for k, v in features.items()})
    return Analysis(features, phases, hmm.log_likelihood(X), agent.reps, feedback, fps, [None] * len(agent.reps))


# ---------------- живая камера ----------------


class ExerciseLiveSession:
    """Как LiveSession для приседа: точки кадра -> признаки (среднее только по прошлым кадрам)
    -> HMM-фильтрация (прямой алгоритм) -> тот же агент, что для видео."""

    def __init__(self, spec: ExerciseSpec, hmm_params: dict | None = None, rules: dict | None = None, window: int = 5):
        self.spec = spec
        self.hmm_params = {**default_hmm(spec), **(hmm_params or {})}
        self.allowed = squat_hmm(**self.hmm_params).trans > 0
        self.agent = ExerciseCoachAgent(spec, rules, fps=30.0)
        self.window = window
        self.times: list[float] = []
        self.landmarks: list[np.ndarray] = []
        self.raw: dict[str, list[float]] = {}
        self.features: dict[str, list[float]] = {}
        self.phases: list[int] = []
        self.feedback: list[Feedback] = []
        self.log_alpha = None
        self.log_likelihood = 0.0
        self.vis_sum = {"L": 0.0, "R": 0.0}
        self.facing = 0.0

    def _fps(self) -> float:
        t = self.times[-60:]
        return (len(t) - 1) / (t[-1] - t[0]) if len(t) > 1 and t[-1] > t[0] else 30.0

    def _avg(self, key: str) -> float:
        """Скользящее среднее по последним ~0.17 с (как 5 кадров при 30 кадр/с)."""
        n = max(1, round(self.window * self._fps() / 30))
        vals = np.array(self.raw[key][-n:])
        vals = vals[~np.isnan(vals)]
        return float(vals.mean()) if len(vals) else np.nan

    def _frame_values(self, pts) -> dict | None:
        if pts is None or np.isnan(pts[:, :2]).any():
            return None
        for side in SIDES:
            self.vis_sum[side] += float(pts[_required_idx(self.spec, side), 3].mean())
        side = max(self.vis_sum, key=self.vis_sum.get)
        if not _visible(pts, self.spec, side):
            return None
        self.facing += self.spec.facing_vote(pts, side)
        return self.spec.values(pts, side, self.facing)

    def push(self, t: float, points) -> dict:
        """Новый кадр: t — время в секундах, points — (33, 4) или None (человека нет)."""
        pts = None if points is None else np.asarray(points, dtype=float).reshape(33, 4)
        self.times.append(float(t))
        self.landmarks.append(np.full((33, 4), np.nan) if pts is None else pts)
        raw = self._frame_values(pts)
        if raw is not None and not self.raw:  # первый кадр с человеком: заводим списки признаков
            before = len(self.times) - 1
            self.raw = {k: [np.nan] * before for k in raw}
            self.features = {k: [np.nan] * before for k in [*raw, "main_vel"]}
        for k in self.raw:
            self.raw[k].append(np.nan if raw is None else raw[k])

        events: list[Feedback] = []
        if raw is None:
            for k in self.features:
                self.features[k].append(np.nan)
            self.phases.append(self.phases[-1] if self.phases else 0)  # человека не видно — фазу не меняем
        else:
            values = {k: self._avg(k) for k in self.raw}
            main = values[self.spec.main]
            hist = self.features[self.spec.main]
            prev = hist[-2] if len(hist) >= 2 else np.nan
            dt = self.times[-1] - self.times[-3] if len(self.times) >= 3 else 0.0
            values["main_vel"] = (main - prev) / dt if dt > 0 and not np.isnan(prev) else 0.0
            for k, v in values.items():
                self.features[k].append(float(v))
            dt_frame = self.times[-1] - self.times[-2] if len(self.times) > 1 else 1 / 30
            self.log_alpha, step_ll = hmm_for_step(self.hmm_params, dt_frame).filter_step(
                self.log_alpha, [main, values["main_vel"]])
            self.log_likelihood += step_ll
            phase = next_phase(self.allowed, self.log_alpha, self.phases[-1] if self.phases else None)
            self.phases.append(phase)
            self.agent.fps = self._fps()
            events = self.agent.step(len(self.times) - 1, phase, values)
            self.feedback += events
        main_now = self.features[self.spec.main][-1] if self.features else np.nan
        return {
            "person": raw is not None,
            "phase": self.phases[-1],
            "knee": None if np.isnan(main_now) else round(float(main_now), 1),  # главный угол (имя поля — как у приседа)
            "reps": len(self.agent.reps),
            "good": sum(r.good for r in self.agent.reps),
            "events": events,
        }

    def summary(self, classifier=None) -> Analysis:
        if len(self.times) < 2 or not self.features:
            raise ValueError("Слишком короткий подход.")
        span = self.times[-1] - self.times[0]
        fps = (len(self.times) - 1) / span if span > 0 else 30.0
        features = {k: np.array(v) for k, v in self.features.items()}
        return Analysis(features, np.array(self.phases), self.log_likelihood, self.agent.reps, self.feedback, fps,
                        [None] * len(self.agent.reps))


# ---------------- демо без видео ----------------


@dataclass
class DeadliftRep:
    hip: float = 90.0  # угол таза внизу
    knee: float = 125.0  # угол колена внизу
    bar: float = 0.25  # наибольшее удаление штанги от стопы
    lean_back: float = -3.0  # наклон назад в момент, когда встал
    seconds: float = 2.5


DEADLIFT_DEMO = {
    "bar_far": DeadliftRep(bar=0.8),
    "lean_back": DeadliftRep(lean_back=25.0),
    "squat_it": DeadliftRep(knee=85.0),
    "stiff_legs": DeadliftRep(knee=172.0),
    "fast": DeadliftRep(seconds=0.8),
}
DEMO_ERRORS = {"deadlift": list(DEADLIFT_DEMO)}


def deadlift_synthetic(reps: list[DeadliftRep], fps: float = 30.0, pause_sec: float = 1.0, noise: float = 1.5,
                       seed: int = 0) -> dict[str, np.ndarray]:
    """Синтетическая становая тяга: признаки, заданные формулой (для демо и тестов)."""
    rng = np.random.default_rng(seed)
    n_pause = int(pause_sec * fps)
    stand = {"hip_angle": 172.0, "knee_angle": 172.0, "lean_back": -3.0, "bar_dist": 0.2}
    parts = {k: [np.full(n_pause, v)] for k, v in stand.items()}
    for r in reps:
        n = max(int(r.seconds * fps), 3)
        k = (1 - np.cos(np.linspace(0, 2 * np.pi, n))) / 2  # 0 -> 1 -> 0: вниз и обратно
        end = np.exp(-(((np.arange(n) / n) - 0.92) / 0.08) ** 2)  # всплеск в конце — момент, когда встал
        parts["hip_angle"].append(172 - (172 - r.hip) * k)
        parts["knee_angle"].append(172 - (172 - r.knee) * k)
        parts["bar_dist"].append(0.2 + (r.bar - 0.2) * k)
        parts["lean_back"].append(-3 - 20 * k + (r.lean_back + 3) * end)
        for key, v in stand.items():
            parts[key].append(np.full(n_pause, v))
    feats = {}
    for key, chunks in parts.items():
        x = np.concatenate(chunks)
        feats[key] = smooth(x + rng.normal(0, 0.01 if key == "bar_dist" else noise, len(x)), 5)
    feats["main_vel"] = np.gradient(feats["hip_angle"]) * fps
    return feats


def demo_session(spec: ExerciseSpec, errors: list[str], fps: float = 30.0) -> dict[str, np.ndarray]:
    """Демо: правильное повторение, потом по очереди выбранные ошибки, между ними — правильные."""
    plan = [DeadliftRep()]
    for e in errors:
        plan += [DEADLIFT_DEMO[e], DeadliftRep()]
    return deadlift_synthetic(plan, fps)
