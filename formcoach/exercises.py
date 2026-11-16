"""Другие упражнения (становая тяга, отжимания, подтягивания, стойка на руках) на тех же AI-темах, что и приседания.

Цепочка та же, меняется только "что измерять":
  точки тела -> признаки упражнения -> HMM по ГЛАВНОМУ углу (неделя 7: ВЕРХ/ОПУСКАНИЕ/НИЗ/ПОДЪЁМ)
             -> агент-тренер с правилами этого упражнения (неделя 1)

Упражнение описывается данными (ExerciseSpec): главный угол, параметры HMM, список проверок
"признак за повторение больше/меньше порога -> ошибка" и тексты на двух языках. Новое упражнение —
это новое описание + функция признаков одного кадра; HMM, агент и живой режим общие.

Стойка на руках — упражнение на удержание: одна "попытка" = вышел в стойку -> держит -> вернулся на ноги.
Для HMM это то же самое, что повторение: ВЕРХ = стоит на ногах, НИЗ = стоит на руках (spec.hold=True).

Приседания остаются в agent.py/live.py: у них есть классификатор (неделя 8) и особые правила (ходьба).
"""

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from . import pose as P
from .agent import Feedback
from .features import FOOT_MIN_VISIBILITY, joint_angle, lean_from_vertical, smooth
from .hmm import BOTTOM, TOP, UP, squat_hmm
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
    hold: bool = False  # упражнение на удержание: ошибки и время считаем только в фазе НИЗ (стойка)
    from_floor: bool = False  # повторение может начинаться снизу (штанга на полу): низ -> верх -> низ
    pose_ok: Callable | None = None  # (точки, сторона) -> bool: поза вообще похожа на это упражнение? Нет — кадр пропускаем
    # (min/max признаков за повторение) -> bool: было ли это настоящим повторением. None — не проверяем
    rep_ok: Callable | None = None
    extra: tuple[str, ...] = ()  # признаки, которые агент копит только для rep_ok (не ошибки и не метрики)
    top_range: tuple[float, float] = (140.0, 180.0)  # пределы ползунков HMM "угол вверху" / "угол внизу"
    bottom_range: tuple[float, float] = (50.0, 130.0)

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
        Field("max_bar_dist", 0.7, 0.2, 1.2, 0.05),  # 0.5 давал 12% ложных замечаний на реальных видео (MyDeadlift)
        Field("max_lean_back", 15.0, 5.0, 40.0, 1.0, "°"),
        Field("min_knee_angle", 90.0, 70.0, 140.0, 1.0, "°"),  # подобран на обучающей части MyDeadlift (было 100)
        Field("max_knee_angle", 165.0, 140.0, 180.0, 1.0, "°"),
        Field("min_rep_seconds", 1.0, 0.3, 3.0, 0.1, " s"),
    ),
    values=deadlift_values,
    facing_vote=deadlift_facing,
    required=("sh", "hip", "knee"),
    from_floor=True,
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
            "name": "Зілтемірді жерден көтеру",
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
                "min_knee_angle": ("Төменде тізе бұрышы, ең аз", "Аз болса — жаттығу отырып-тұруға айналады."),
                "max_knee_angle": ("Төменде тізе бұрышы, ең көп", "Көп болса — аяқ тым түзу."),
                "min_rep_seconds": ("Қайталаудың ең аз уақыты", "Жылдамырақ — штанганы бақылаусыз «тастайды»."),
            },
        },
    },
)


# ---------------- отжимания ----------------


def pushup_values(pts: np.ndarray, side: str, facing: float) -> dict[str, float]:
    """Признаки отжимания на одном кадре (камера сбоку, на уровне пола).

    elbow_angle — угол в локте (плечо-локоть-кисть): ~160+ на прямых руках, ~90 внизу. Это главный угол.
    sag  — насколько таз провис НИЖЕ линии плечо-лодыжка, градусы (0 — тело прямое или таз выше).
    pike — насколько таз задран ВЫШЕ этой линии, градусы.
    Изгиб тела = 180 - угол плечо-таз-лодыжка; выше или ниже линии — смотрим по координате y таза.
    """
    s = SIDES[side]
    xy = pts[:, :2]
    foot = s["ankle"] if pts[s["ankle"], 3] >= MIN_VISIBILITY else s["knee"]  # лодыжку не видно — берём колено
    sh, hip, end = xy[s["sh"]], xy[s["hip"]], xy[foot]
    bend = 180.0 - float(joint_angle(sh, hip, end))
    dx = end[0] - sh[0]
    line_y = sh[1] + (end[1] - sh[1]) * (hip[0] - sh[0]) / dx if abs(dx) > 1e-6 else hip[1]
    below = hip[1] > line_y  # ось y вниз: больше y — ниже на картинке
    return {
        "elbow_angle": float(joint_angle(xy[s["sh"]], xy[s["el"]], xy[s["wr"]])),
        "sag": bend if below else 0.0,
        "pike": 0.0 if below else bend,
    }


MAX_PLANK_TILT = 45.0  # линия плечо-стопа в упоре лёжа почти горизонтальна (~20–35°); стоя или в наклоне — 60–90°


def in_plank(pts: np.ndarray, side: str) -> bool:
    """Человек в упоре лёжа: линия плечо -> лодыжка (или колено) наклонена к горизонту меньше чем на 45°.
    Стоя, при ходьбе или в наклоне локоть тоже сгибается — без этой проверки такие кадры считались отжиманиями
    (нашли на живом тесте с телефона 2026-11-16)."""
    s = SIDES[side]
    foot = s["ankle"] if pts[s["ankle"], 3] >= MIN_VISIBILITY else s["knee"]
    dx, dy = pts[foot, :2] - pts[s["sh"], :2]
    return bool(np.degrees(np.arctan2(abs(dy), abs(dx))) < MAX_PLANK_TILT)


def no_facing(pts: np.ndarray, side: str) -> float:
    """Для отжиманий направление взгляда не нужно."""
    return 0.0


PUSHUP = ExerciseSpec(
    key="pushup",
    main="elbow_angle",
    top=160.0,
    bottom=90.0,
    speed=90.0,
    min_dip=30.0,  # неглубокое отжимание (локоть до ~125°) тоже засчитываем — с ошибкой "неглубоко"
    checks=(
        Check("shallow", "elbow_angle", "min", ">", "max_bottom_elbow"),
        Check("sag", "sag", "max", ">", "max_sag", live=True),
        Check("pike", "pike", "max", ">", "max_pike", live=True),
    ),
    fields=(
        Field("max_bottom_elbow", 100.0, 70.0, 130.0, 1.0, "°"),
        Field("max_sag", 15.0, 5.0, 40.0, 1.0, "°"),
        Field("max_pike", 20.0, 5.0, 45.0, 1.0, "°"),
        Field("min_rep_seconds", 0.8, 0.3, 3.0, 0.1, " s"),
    ),
    values=pushup_values,
    facing_vote=no_facing,
    pose_ok=in_plank,
    required=("sh", "el", "wr", "hip"),
    metrics=(("elbow_angle", "min", "°"), ("sag", "max", "°"), ("pike", "max", "°")),
    text={
        "ru": {
            "name": "Отжимания",
            "phases": ["ВЕРХ", "ОПУСКАНИЕ", "НИЗ", "ПОДЪЁМ"],
            "error": {
                "shallow": "Неглубоко — опускайся, пока локти не согнутся до 90°",
                "sag": "Таз провисает — напряги пресс и ягодицы, тело в одну линию",
                "pike": "Таз задран вверх — опусти его, тело в одну линию",
                "fast": "Слишком быстро — опускайся подконтрольно",
            },
            "short": {"shallow": "неглубоко", "sag": "провис", "pike": "таз вверх", "fast": "быстро"},
            "live": {"sag": "Не прогибайся в пояснице", "pike": "Опусти таз"},
            "metric": {"elbow_angle": "Локоть", "sag": "Провис", "pike": "Таз вверх"},
            "fields": {
                "max_bottom_elbow": ("Угол локтя внизу, макс.", "Больше — отжимание неглубокое. 90° — локоть под прямым углом."),
                "max_sag": ("Провис таза, макс.", "На сколько градусов таз может уйти ниже линии плечо–лодыжка."),
                "max_pike": ("Таз вверх, макс.", "На сколько градусов таз может подняться выше этой линии."),
                "min_rep_seconds": ("Мин. время повторения", "Быстрее — «падает» вниз без контроля."),
            },
        },
        "kk": {
            "name": "Отжимание",
            "phases": ["ЖОҒАРЫ", "ТҮСУ", "ТӨМЕН", "КӨТЕРІЛУ"],
            "error": {
                "shallow": "Таяз — шынтақ 90°-қа дейін бүгілгенше төмен түс",
                "sag": "Жамбас төмен салбырайды — іш пен бөксені қатайт, дене бір түзу сызықта болсын",
                "pike": "Жамбас жоғары көтерілген — оны түсір, дене бір түзу сызықта болсын",
                "fast": "Тым жылдам — бақылап, баяу түс",
            },
            "short": {"shallow": "таяз", "sag": "салбырау", "pike": "жамбас жоғары", "fast": "жылдам"},
            "live": {"sag": "Белді майыстырма", "pike": "Жамбасты түсір"},
            "metric": {"elbow_angle": "Шынтақ", "sag": "Салбырау", "pike": "Жамбас жоғары"},
            "fields": {
                "max_bottom_elbow": ("Төменде шынтақ бұрышы, ең көп", "Көп болса — таяз отжимание. 90° — шынтақ тік бұрышпен."),
                "max_sag": ("Жамбастың салбырауы, ең көп", "Жамбас иық–тобық сызығынан неше градус төмен түсе алады."),
                "max_pike": ("Жамбас жоғары, ең көп", "Жамбас бұл сызықтан неше градус жоғары көтеріле алады."),
                "min_rep_seconds": ("Қайталаудың ең аз уақыты", "Жылдамырақ — бақылаусыз төмен «құлайды»."),
            },
        },
    },
)


# ---------------- подтягивания на турнике ----------------


def pullup_values(pts: np.ndarray, side: str, facing: float) -> dict[str, float]:
    """Признаки подтягивания на одном кадре (лучше сбоку; спереди не видно раскачку).

    elbow_angle — угол в локте: ~165+ в полном висе, ~40 наверху. Это главный угол.
    chin_gap — насколько рот НИЖЕ кистей (перекладины), в длинах предплечья: <0 — подбородок над перекладиной.
    lean — наклон корпуса (плечо-таз) от вертикали: раскачка и рывки.
    """
    s = SIDES[side]
    xy = pts[:, :2]
    forearm = max(float(np.linalg.norm(xy[s["el"]] - xy[s["wr"]])), 1e-6)
    mouth_vis = min(pts[P.MOUTH_L, 3], pts[P.MOUTH_R, 3])
    mouth_y = (xy[P.MOUTH_L, 1] + xy[P.MOUTH_R, 1]) / 2
    chin = float((mouth_y - xy[s["wr"], 1]) / forearm) if mouth_vis >= MIN_VISIBILITY else np.nan
    lean = float(lean_from_vertical(xy[s["sh"]], xy[s["hip"]])) if pts[s["hip"], 3] >= MIN_VISIBILITY else np.nan
    return {
        "elbow_angle": float(joint_angle(xy[s["sh"]], xy[s["el"]], xy[s["wr"]])),
        "chin_gap": chin,
        "lean": lean,
    }


def on_bar(pts: np.ndarray, side: str) -> bool:
    """Человек держится за перекладину: обе кисти выше плеч (y в картинке растёт вниз).
    Пока он подходит к турнику, отдыхает или машет руками, локоть тоже сгибается — такие кадры
    не должны считаться подтягиваниями. Правило появилось после проверки на видео RepCount (scripts/eval_real.py)."""
    y = pts[:, 1]
    return bool(max(y[P.L_WRIST], y[P.R_WRIST]) < min(y[P.L_SHOULDER], y[P.R_SHOULDER]))


PULLUP = ExerciseSpec(
    key="pullup",
    main="elbow_angle",
    top=165.0,
    bottom=60.0,
    speed=120.0,
    min_dip=40.0,
    checks=(
        Check("chin_low", "chin_gap", "min", ">", "max_chin_gap"),
        Check("no_hang", "main_start", "max", "<", "min_hang_elbow"),
        Check("swing", "lean", "max", ">", "max_swing", live=True),
    ),
    fields=(
        Field("max_chin_gap", 0.0, -0.5, 0.5, 0.05),
        Field("min_hang_elbow", 150.0, 120.0, 175.0, 1.0, "°"),
        Field("max_swing", 25.0, 10.0, 50.0, 1.0, "°"),
        Field("min_rep_seconds", 1.0, 0.3, 3.0, 0.1, " s"),
    ),
    values=pullup_values,
    facing_vote=no_facing,
    required=("sh", "el", "wr"),
    pose_ok=on_bar,
    metrics=(("elbow_angle", "min", "°"), ("chin_gap", "min", ""), ("main_start", "max", "°"), ("lean", "max", "°")),
    text={
        "ru": {
            "name": "Подтягивания",
            # главный угол уменьшается, когда человек поднимается, поэтому "ВЕРХ" HMM — это вис
            "phases": ["ВИС", "ПОДЪЁМ", "ВВЕРХУ", "ОПУСКАНИЕ"],
            "error": {
                "chin_low": "Подбородок не поднялся выше перекладины — тянись до конца",
                "no_hang": "Руки не выпрямились внизу — опускайся до полного виса",
                "swing": "Раскачиваешься — держи корпус ровно, без рывков",
                "fast": "Слишком быстро — опускайся подконтрольно",
            },
            "short": {"chin_low": "подбородок", "no_hang": "не довис", "swing": "раскачка", "fast": "быстро"},
            "live": {"swing": "Не раскачивайся"},
            "metric": {"elbow_angle": "Локоть", "chin_gap": "Подбородок", "main_start": "Вис", "lean": "Раскачка"},
            "fields": {
                "max_chin_gap": ("Рот относительно перекладины", "В длинах предплечья: 0 — рот на уровне кистей, меньше 0 — выше перекладины."),
                "min_hang_elbow": ("Угол локтя в висе, мин.", "Меньше — руки не выпрямились, полуповторение."),
                "max_swing": ("Раскачка корпуса, макс.", "Наклон корпуса от вертикали. Видно только сбоку."),
                "min_rep_seconds": ("Мин. время повторения", "Быстрее — рывками, без контроля."),
            },
        },
        "kk": {
            "name": "Белтемірде тартылу",
            "phases": ["АСЫЛУ", "КӨТЕРІЛУ", "ЖОҒАРЫДА", "ТҮСУ"],
            "error": {
                "chin_low": "Иек турниктен жоғары көтерілмеді — соңына дейін тартыл",
                "no_hang": "Төменде қол түзелмеді — толық асылғанша түс",
                "swing": "Тербелесің — денені тік ұста, серпілмей",
                "fast": "Тым жылдам — бақылап, баяу түс",
            },
            "short": {"chin_low": "иек", "no_hang": "толық асылмау", "swing": "тербелу", "fast": "жылдам"},
            "live": {"swing": "Тербелме"},
            "metric": {"elbow_angle": "Шынтақ", "chin_gap": "Иек", "main_start": "Асылу", "lean": "Тербелу"},
            "fields": {
                "max_chin_gap": ("Ауыздың турникке қатысты орны", "Білек ұзындығымен: 0 — ауыз қол деңгейінде, 0-ден аз — турниктен жоғары."),
                "min_hang_elbow": ("Асылғанда шынтақ бұрышы, ең аз", "Аз болса — қол түзелмеді, жарты қайталау."),
                "max_swing": ("Дененің тербелуі, ең көп", "Дененің тік сызықтан ауытқуы. Тек бүйірден көрінеді."),
                "min_rep_seconds": ("Қайталаудың ең аз уақыты", "Жылдамырақ — серпіліп, бақылаусыз."),
            },
        },
    },
)

# ---------------- брусья ----------------


def dips_values(pts: np.ndarray, side: str, facing: float) -> dict[str, float]:
    """Признаки отжиманий на брусьях на одном кадре (камера сбоку).

    elbow_angle   — угол в локте: ~165+ наверху на прямых руках, ~90 внизу. Это главный угол.
    shoulder_drop — насколько плечо опустилось НИЖЕ локтя, в длинах плеча (<0 — плечо выше локтя).
                    Слишком глубоко (плечо сильно ниже локтя) — большая нагрузка на плечевой сустав.
    sh_y, wr_y    — высота плеча и кисти в длинах плеча: на брусьях кисти стоят на месте, а двигается тело.
                    Если наоборот (двигаются кисти) — человек просто сгибает руки стоя, это не повторение.
    """
    s = SIDES[side]
    xy = pts[:, :2]
    upper = max(float(np.linalg.norm(xy[s["sh"]] - xy[s["el"]])), 1e-6)
    return {
        "elbow_angle": float(joint_angle(xy[s["sh"]], xy[s["el"]], xy[s["wr"]])),
        "shoulder_drop": float((xy[s["sh"], 1] - xy[s["el"], 1]) / upper),
        "sh_y": float(xy[s["sh"], 1] / upper),
        "wr_y": float(xy[s["wr"], 1] / upper),
    }


MAX_DIPS_TILT = 50.0  # на брусьях корпус ближе к вертикали; лёжа (отжимания, жим) — почти горизонтально


def on_bars(pts: np.ndarray, side: str) -> bool:
    """Поза похожа на брусья: кисти ниже плеч (упор на руки), корпус скорее вертикальный, чем лёжа."""
    s = SIDES[side]
    y = pts[:, 1]
    hands_low = max(y[P.L_SHOULDER], y[P.R_SHOULDER]) < min(y[P.L_WRIST], y[P.R_WRIST])
    return bool(hands_low and lean_from_vertical(pts[s["sh"], :2], pts[s["hip"], :2]) < MAX_DIPS_TILT)


def body_moves(agg: dict) -> bool:
    """За повторение двигалось тело, а кисти стояли на брусьях: размах кисти меньше половины размаха плеча.
    Стоя у брусьев человек тоже может сгибать руки — тогда двигаются кисти, а плечи на месте."""
    sh = agg.get("sh_y:max", np.nan) - agg.get("sh_y:min", np.nan)
    wr = agg.get("wr_y:max", np.nan) - agg.get("wr_y:min", np.nan)
    return bool(sh >= 0.3 and wr <= 0.5 * sh)


DIPS = ExerciseSpec(
    key="dips",
    main="elbow_angle",
    top=165.0,
    bottom=90.0,
    speed=100.0,
    min_dip=35.0,  # неглубокое повторение (локоть до ~130°) тоже считаем — с ошибкой "неглубоко"
    checks=(
        Check("shallow", "elbow_angle", "min", ">", "max_bottom_elbow"),
        Check("too_deep", "shoulder_drop", "max", ">", "max_shoulder_drop", live=True),
        Check("no_lockout", "main_start", "max", "<", "min_lock_elbow"),
    ),
    fields=(
        Field("max_bottom_elbow", 100.0, 70.0, 130.0, 1.0, "°"),
        Field("max_shoulder_drop", 0.35, 0.0, 1.0, 0.05),
        Field("min_lock_elbow", 150.0, 120.0, 175.0, 1.0, "°"),
        Field("min_rep_seconds", 1.0, 0.3, 3.0, 0.1, " s"),
    ),
    values=dips_values,
    facing_vote=no_facing,
    required=("sh", "el", "wr", "hip"),
    pose_ok=on_bars,
    rep_ok=body_moves,
    extra=("sh_y", "wr_y"),
    metrics=(("elbow_angle", "min", "°"), ("shoulder_drop", "max", ""), ("main_start", "max", "°")),
    text={
        "ru": {
            "name": "Брусья",
            "phases": ["ВВЕРХУ", "ОПУСКАНИЕ", "ВНИЗУ", "ПОДЪЁМ"],
            "error": {
                "shallow": "Неглубоко — опускайся, пока локти не согнутся до 90°",
                "too_deep": "Слишком глубоко — плечи сильно ниже локтей, это вредно для плечевых суставов",
                "no_lockout": "Руки не выпрямились наверху — выжимай до прямых локтей",
                "fast": "Слишком быстро — опускайся подконтрольно",
            },
            "short": {"shallow": "неглубоко", "too_deep": "слишком глубоко", "no_lockout": "не выпрямил", "fast": "быстро"},
            "live": {"too_deep": "Не опускайся так низко"},
            "metric": {"elbow_angle": "Локоть", "shoulder_drop": "Плечо ниже локтя", "main_start": "Наверху"},
            "fields": {
                "max_bottom_elbow": ("Угол локтя внизу, макс.", "Больше — неглубокое повторение. 90° — локоть под прямым углом."),
                "max_shoulder_drop": ("Плечо ниже локтя, макс.", "В длинах плеча: 0 — плечо на уровне локтя."),
                "min_lock_elbow": ("Угол локтя наверху, мин.", "Меньше — руки не выпрямились, полуповторение."),
                "min_rep_seconds": ("Мин. время повторения", "Быстрее — «падение» вниз без контроля."),
            },
        },
        "kk": {
            "name": "Брусья",
            "phases": ["ЖОҒАРЫДА", "ТҮСУ", "ТӨМЕНДЕ", "КӨТЕРІЛУ"],
            "error": {
                "shallow": "Таяз — шынтақ 90°-қа бүгілгенше түс",
                "too_deep": "Тым терең — иық шынтақтан әлдеқайда төмен, бұл иық буынына зиян",
                "no_lockout": "Жоғарыда қол түзелмеді — шынтақ түзелгенше көтеріл",
                "fast": "Тым жылдам — бақылап, баяу түс",
            },
            "short": {"shallow": "таяз", "too_deep": "тым терең", "no_lockout": "түзелмеді", "fast": "жылдам"},
            "live": {"too_deep": "Тым төмен түспе"},
            "metric": {"elbow_angle": "Шынтақ", "shoulder_drop": "Иық шынтақтан төмен", "main_start": "Жоғарыда"},
            "fields": {
                "max_bottom_elbow": ("Төменде шынтақ бұрышы, ең көп", "Көп болса — таяз қайталау. 90° — шынтақ тік бұрышпен."),
                "max_shoulder_drop": ("Иық шынтақтан төмен, ең көп", "Иық ұзындығымен: 0 — иық шынтақ деңгейінде."),
                "min_lock_elbow": ("Жоғарыда шынтақ бұрышы, ең аз", "Аз болса — қол түзелмеді, жарты қайталау."),
                "min_rep_seconds": ("Қайталаудың ең аз уақыты", "Жылдамырақ — бақылаусыз төмен «құлайды»."),
            },
        },
    },
)


# ---------------- жим лёжа ----------------


def bench_values(pts: np.ndarray, side: str, facing: float) -> dict[str, float]:
    """Признаки жима лёжа на одном кадре (камера сбоку, на уровне скамьи).

    elbow_angle — угол в локте: ~160+ на прямых руках наверху, ~80 когда гриф у груди. Это главный угол.
    hip_rise    — насколько таз ВЫШЕ плеча, в длинах корпуса: таз оторвался от скамьи (лёжа ровно — около 0).
    """
    s = SIDES[side]
    xy = pts[:, :2]
    torso = max(float(np.linalg.norm(xy[s["sh"]] - xy[s["hip"]])), 1e-6)
    return {
        "elbow_angle": float(joint_angle(xy[s["sh"]], xy[s["el"]], xy[s["wr"]])),
        "hip_rise": float((xy[s["sh"], 1] - xy[s["hip"], 1]) / torso),
    }


MAX_BENCH_TILT = 40.0  # лёжа на скамье линия плечо-таз почти горизонтальна


def on_bench(pts: np.ndarray, side: str) -> bool:
    """Человек лежит, руки над грудью: корпус почти горизонтален и кисти выше плеч.
    Стоя (корпус вертикален) или в упоре лёжа (кисти ниже плеч) — не жим."""
    s = SIDES[side]
    y = pts[:, 1]
    hands_up = (y[P.L_WRIST] + y[P.R_WRIST]) / 2 < (y[P.L_SHOULDER] + y[P.R_SHOULDER]) / 2
    dx, dy = pts[s["hip"], :2] - pts[s["sh"], :2]
    return bool(hands_up and np.degrees(np.arctan2(abs(dy), abs(dx))) < MAX_BENCH_TILT)


BENCH = ExerciseSpec(
    key="bench",
    main="elbow_angle",
    top=160.0,
    bottom=85.0,
    speed=100.0,
    min_dip=35.0,
    checks=(
        Check("no_chest", "elbow_angle", "min", ">", "max_bottom_elbow"),
        Check("butt_up", "hip_rise", "max", ">", "max_hip_rise", live=True),
        Check("no_lockout", "main_start", "max", "<", "min_lock_elbow"),
    ),
    fields=(
        Field("max_bottom_elbow", 100.0, 70.0, 130.0, 1.0, "°"),
        Field("max_hip_rise", 0.2, 0.05, 0.6, 0.05),
        Field("min_lock_elbow", 150.0, 120.0, 175.0, 1.0, "°"),
        Field("min_rep_seconds", 1.0, 0.3, 3.0, 0.1, " s"),
    ),
    values=bench_values,
    facing_vote=no_facing,
    required=("sh", "el", "wr", "hip"),
    pose_ok=on_bench,
    metrics=(("elbow_angle", "min", "°"), ("hip_rise", "max", ""), ("main_start", "max", "°")),
    text={
        "ru": {
            "name": "Жим лёжа",
            "phases": ["ВВЕРХУ", "ОПУСКАНИЕ", "У ГРУДИ", "ЖИМ"],
            "error": {
                "no_chest": "Неполная амплитуда — опускай гриф до груди",
                "butt_up": "Таз оторвался от скамьи — держи ягодицы на скамье",
                "no_lockout": "Руки не выпрямились наверху — выжимай до прямых локтей",
                "fast": "Слишком быстро — не бросай гриф на грудь, опускай подконтрольно",
            },
            "short": {"no_chest": "не до груди", "butt_up": "таз оторван", "no_lockout": "не выпрямил", "fast": "быстро"},
            "live": {"butt_up": "Таз на скамью"},
            "metric": {"elbow_angle": "Локоть", "hip_rise": "Таз выше плеч", "main_start": "Наверху"},
            "fields": {
                "max_bottom_elbow": ("Угол локтя внизу, макс.", "Больше — гриф не дошёл до груди."),
                "max_hip_rise": ("Таз выше плеч, макс.", "В длинах корпуса: лёжа ровно — около 0."),
                "min_lock_elbow": ("Угол локтя наверху, мин.", "Меньше — руки не выпрямились, полуповторение."),
                "min_rep_seconds": ("Мин. время повторения", "Быстрее — гриф «бросают» и отбивают от груди."),
            },
        },
        "kk": {
            "name": "Жатып итеру",
            "phases": ["ЖОҒАРЫДА", "ТҮСІРУ", "КЕУДЕДЕ", "ИТЕРУ"],
            "error": {
                "no_chest": "Толық емес — грифті кеудеге дейін түсір",
                "butt_up": "Жамбас орындықтан көтерілді — бөксені орындықта ұста",
                "no_lockout": "Жоғарыда қол түзелмеді — шынтақ түзелгенше итер",
                "fast": "Тым жылдам — грифті кеудеге тастама, бақылап түсір",
            },
            "short": {"no_chest": "кеудеге жетпеді", "butt_up": "жамбас көтерілді", "no_lockout": "түзелмеді", "fast": "жылдам"},
            "live": {"butt_up": "Жамбасты орындыққа"},
            "metric": {"elbow_angle": "Шынтақ", "hip_rise": "Жамбас иықтан жоғары", "main_start": "Жоғарыда"},
            "fields": {
                "max_bottom_elbow": ("Төменде шынтақ бұрышы, ең көп", "Көп болса — гриф кеудеге жетпеді."),
                "max_hip_rise": ("Жамбас иықтан жоғары, ең көп", "Дене ұзындығымен: түзу жатқанда — шамамен 0."),
                "min_lock_elbow": ("Жоғарыда шынтақ бұрышы, ең аз", "Аз болса — қол түзелмеді, жарты қайталау."),
                "min_rep_seconds": ("Қайталаудың ең аз уақыты", "Жылдамырақ — грифті «тастап», кеудеден серпеді."),
            },
        },
    },
)

EXERCISES: dict[str, ExerciseSpec] = {"deadlift": DEADLIFT, "pushup": PUSHUP, "pullup": PULLUP, "dips": DIPS, "bench": BENCH}


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

    Исходное положение (self._home) — то, в котором человек отдыхает между повторениями. Обычно это ВЕРХ
    (стоя). Но классическую становую тягу начинают с пола: если видео началось в фазе НИЗ и упражнение
    это разрешает (spec.from_floor), исходным считается НИЗ, а повторение — низ -> верх -> низ.
    Так решено после проверки на реальных видео (scripts/eval_real.py, датасет MyDeadlift).
    """

    def __init__(self, spec: ExerciseSpec, rules: dict | None = None, fps: float = 30.0):
        self.spec = spec
        self.rules = {**spec.default_rules(), **(rules or {})}
        self.fps = fps
        self.reps: list[ExerciseRep] = []
        self._in_rep = False
        self._home: int | None = None  # исходное положение: TOP или BOTTOM, выбирается по началу видео
        self._standing = -np.inf  # самый большой главный угол стоя перед повторением
        self._warned: set[str] = set()
        self._last_end = -10**9  # кадр конца последнего повторения
        names = {spec.main} | {c.feature for c in spec.checks} | {m[0] for m in spec.metrics} | set(spec.extra)
        # main_start и hold_seconds — не признаки кадра, их считает сам агент
        self._tracked = sorted(names - {"main_start", "hold_seconds"})

    def _start_rep(self, frame: int, v: dict):
        self._in_rep, self._start, self._warned, self._agg = True, frame, set(), {}
        self._bottom_frame, self._min_main = frame, v[self.spec.main]
        self._max_main = v[self.spec.main]
        self._seen = 0.0  # на скольких кадрах повторения человек был виден в подходящей позе
        self._hold_frames = 0  # сколько кадров подряд в фазе НИЗ (для упражнений на удержание)
        self._update(frame, v, TOP)

    HOLD_GRACE_SEC = 0.3  # первые доли секунды в стойке тело ещё выравнивается после маха — не ругаем
    HOLD_MAX_TILT = 30.0  # и оцениваем, только когда тело почти вертикально (главный угол <= "внизу" + 30)
    HOLD_MAX_SPEED = 45.0  # и почти неподвижно, град/с: в начале спуска колени сгибаются — это не ошибка

    def _judged(self, phase: int, v: dict) -> bool:
        """Оценивать ли этот кадр. Для повторений — всегда. Для удержания — только когда человек
        уже стоит в стойке (фаза НИЗ) дольше HOLD_GRACE_SEC, почти вертикально и не двигается:
        мах и спуск не оцениваем."""
        if not self.spec.hold:
            return True
        return (phase == BOTTOM and self._hold_frames / self.fps > self.HOLD_GRACE_SEC
                and v[self.spec.main] <= self.spec.bottom + self.HOLD_MAX_TILT
                and abs(v.get("main_vel", 0.0)) <= self.HOLD_MAX_SPEED)

    def _update(self, frame: int, v: dict, phase: int):
        if v[self.spec.main] < self._min_main:
            self._min_main, self._bottom_frame = v[self.spec.main], frame
        self._max_main = max(self._max_main, v[self.spec.main])
        self._seen += v.get("seen", 1.0)  # в видео пропуски заполнены соседними значениями и помечены seen=0
        if self.spec.hold:
            self._hold_frames = self._hold_frames + 1 if phase == BOTTOM else 0
            self._agg["hold_seconds:max"] = max(self._agg.get("hold_seconds:max", 0.0), self._hold_frames / self.fps)
            if not self._judged(phase, v):
                return
        for name in self._tracked:
            x = v.get(name, np.nan)
            self._agg[f"{name}:min"] = np.fmin(self._agg.get(f"{name}:min", np.nan), x)  # fmin/fmax пропускают NaN
            self._agg[f"{name}:max"] = np.fmax(self._agg.get(f"{name}:max", np.nan), x)

    MIN_SEEN = 0.6  # повторение засчитываем, если человек был виден в нужной позе хотя бы на 60% его кадров
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
        if self._home is None:  # начало: откуда человек начинает — стоя или с пола
            if not self.spec.from_floor or phase == TOP:
                self._home = TOP
            elif phase in (BOTTOM, UP):  # у пола или уже отрывает штангу — значит, начали с пола
                self._home = BOTTOM
            else:  # ОПУСКАНИЕ в самом начале: человек ещё наклоняется к штанге — ждём, пока возьмётся
                return actions
        if not self._in_rep:
            if phase == self._home:
                if phase == TOP:
                    self._standing = max(self._standing, v[self.spec.main])
            else:
                self._start_rep(frame, v)
            return actions

        self._update(frame, v, phase)
        for c in self.spec.checks:
            if not self._judged(phase, v):
                break
            if c.live and c.code not in self._warned and c.broken(v.get(c.feature, np.nan), self.rules[c.rule]):
                self._warned.add(c.code)
                actions.append(Feedback(frame, "warn", self.spec.text["ru"]["live"][c.code], len(self.reps) + 1, c.code))

        if phase == self._home:  # вернулся в исходное положение — повторение закончилось
            self._in_rep = False
            long_enough = (frame - self._start) / self.fps >= 0.4
            # размах главного угла за повторение: от положения стоя (или верхней точки, если начали с пола) до низа
            top = self._standing if self._home == TOP else self._max_main
            deep_enough = top - self._min_main >= self.spec.min_dip
            # в живом режиме кадры без человека до агента не доходят, поэтому делим на длину повторения в кадрах
            # (только для упражнений с проверкой позы pose_ok: у остальных плохая видимость сустава — не повод не считать)
            seen_enough = self.spec.pose_ok is None or self._seen / (frame - self._start + 1) >= self.MIN_SEEN
            real = self.spec.rep_ok is None or self.spec.rep_ok(self._agg)  # например, на брусьях двигалось тело, а не кисти
            if long_enough and deep_enough and seen_enough and real:  # иначе дрожание, человека не было видно или не то движение
                actions.append(self._finish(frame))
                self._last_end = frame
            if phase == TOP:
                self._standing = v[self.spec.main]
        return actions

    def _finish(self, end: int) -> Feedback:
        values = {k: float(x) for k, x in self._agg.items()}
        # главный угол в положении стоя перед повторением (если начали с пола — верхняя точка самого повторения)
        values["main_start:max"] = float(self._standing if self._home == TOP else self._max_main)
        rep = ExerciseRep(len(self.reps) + 1, self._start, end, self._bottom_frame, (end - self._start) / self.fps, values)
        rep.errors = [c.code for c in self.spec.checks
                      if c.broken(values.get(f"{c.feature}:{c.agg}", np.nan), self.rules[c.rule])]
        if "min_rep_seconds" in self.rules and rep.seconds < self.rules["min_rep_seconds"]:  # у удержания этого правила нет
            rep.errors.append("fast")
        self.reps.append(rep)
        ru = self.spec.text["ru"]
        if rep.good:
            return Feedback(end, "ok", ru.get("rep_ok", "Повторение {n}: отлично!").format(n=rep.number), rep.number, "rep")
        text = "; ".join(ru["error"][e] for e in rep.errors)
        return Feedback(end, "warn", ru.get("rep_bad", "Повторение {n}: {text}").format(n=rep.number, text=text),
                        rep.number, "rep")


# ---------------- анализ целого видео ----------------


def _required_idx(spec: ExerciseSpec, side: str) -> list[int]:
    return [SIDES[side][j] for j in spec.required]


def _visible(pts: np.ndarray, spec: ExerciseSpec, side: str) -> bool:
    if np.isnan(pts[:, :2]).any() or pts[_required_idx(spec, side), 3].min() < MIN_VISIBILITY:
        return False
    return spec.pose_ok is None or spec.pose_ok(pts, side)


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
    feats["seen"] = np.array(seen, dtype=float)  # 1 — кадр настоящий, 0 — значения взяты у соседних кадров
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
    "bar_far": DeadliftRep(bar=0.95),
    "lean_back": DeadliftRep(lean_back=25.0),
    "squat_it": DeadliftRep(knee=78.0),
    "stiff_legs": DeadliftRep(knee=172.0),
    "fast": DeadliftRep(seconds=0.8),
}


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


@dataclass
class PushupRep:
    elbow: float = 85.0  # угол локтя внизу
    sag: float = 3.0  # провис таза внизу, градусы
    pike: float = 0.0  # таз вверх внизу, градусы
    seconds: float = 2.0


PUSHUP_DEMO = {
    "shallow": PushupRep(elbow=125.0),
    "sag": PushupRep(sag=25.0),
    "pike": PushupRep(pike=30.0),
    "fast": PushupRep(seconds=0.6),
}


def pushup_synthetic(reps: list[PushupRep], fps: float = 30.0, pause_sec: float = 1.0, noise: float = 1.5,
                     seed: int = 0) -> dict[str, np.ndarray]:
    """Синтетические отжимания: признаки, заданные формулой (для демо и тестов)."""
    rng = np.random.default_rng(seed)
    n_pause = int(pause_sec * fps)
    top = 165.0
    parts = {"elbow_angle": [np.full(n_pause, top)], "sag": [np.zeros(n_pause)], "pike": [np.zeros(n_pause)]}
    for r in reps:
        k = (1 - np.cos(np.linspace(0, 2 * np.pi, max(int(r.seconds * fps), 3)))) / 2
        parts["elbow_angle"] += [top - (top - r.elbow) * k, np.full(n_pause, top)]
        parts["sag"] += [r.sag * k, np.zeros(n_pause)]
        parts["pike"] += [r.pike * k, np.zeros(n_pause)]
    feats = {}
    for key, chunks in parts.items():
        x = np.concatenate(chunks) + rng.normal(0, noise, sum(map(len, chunks)))
        feats[key] = smooth(np.maximum(x, 0) if key != "elbow_angle" else x, 5)
    feats["main_vel"] = np.gradient(feats["elbow_angle"]) * fps
    return feats


@dataclass
class PullupRep:
    elbow: float = 40.0  # угол локтя наверху
    chin: float = -0.15  # рот относительно перекладины наверху (<0 — выше)
    hang: float = 170.0  # угол локтя в висе ПЕРЕД этим повторением
    swing: float = 8.0  # наибольший наклон корпуса
    seconds: float = 2.2


PULLUP_DEMO = {
    "chin_low": PullupRep(elbow=70.0, chin=0.35),
    "no_hang": PullupRep(hang=140.0),
    "swing": PullupRep(swing=40.0),
    "fast": PullupRep(seconds=0.7),
}


def pullup_synthetic(reps: list[PullupRep], fps: float = 30.0, pause_sec: float = 1.0, noise: float = 1.5,
                     seed: int = 0) -> dict[str, np.ndarray]:
    """Синтетические подтягивания: признаки, заданные формулой (для демо и тестов).
    Каждое повторение: вис (угол r.hang) -> вверх до r.elbow -> вниз до виса следующего повторения."""
    rng = np.random.default_rng(seed)
    n_pause = int(pause_sec * fps)
    elbow, lean = [], []
    for i, r in enumerate(reps):
        nxt = reps[i + 1].hang if i + 1 < len(reps) else 170.0
        k = (1 - np.cos(np.linspace(0, 2 * np.pi, max(int(r.seconds * fps), 3)))) / 2
        base = np.where(np.arange(len(k)) < len(k) // 2, r.hang, nxt)  # откуда поднимается / куда опускается
        elbow += [np.full(n_pause, r.hang), base - (base - r.elbow) * k]
        lean += [np.full(n_pause, 5.0), 5 + (r.swing - 5) * k]
    elbow.append(np.full(n_pause, 170.0))
    lean.append(np.full(n_pause, 5.0))
    elbow_angle = np.concatenate(elbow)
    # рот поднимается вместе со сгибанием локтя: в висе (170°) он на 2 предплечья ниже перекладины
    tops = np.concatenate([np.full(n_pause + max(int(r.seconds * fps), 3), r.chin) for r in reps] + [np.full(n_pause, -0.15)])
    lows = np.concatenate([np.full(n_pause + max(int(r.seconds * fps), 3), r.elbow) for r in reps] + [np.full(n_pause, 40.0)])
    chin = tops + (2.0 - tops) * (elbow_angle - lows) / (170.0 - lows)
    feats = {
        "elbow_angle": smooth(elbow_angle + rng.normal(0, noise, len(elbow_angle)), 5),
        "chin_gap": smooth(chin + rng.normal(0, 0.02, len(chin)), 5),
        "lean": smooth(np.concatenate(lean) + rng.normal(0, noise, len(elbow_angle)), 5),
    }
    feats["main_vel"] = np.gradient(feats["elbow_angle"]) * fps
    return feats


def _press_elbow(reps: list, fps: float, n_pause: int, top: float = 168.0) -> tuple[np.ndarray, list[np.ndarray]]:
    """Главный угол для брусьев и жима: наверху угол r.lock (прямые руки) -> вниз до r.elbow -> наверх до lock
    следующего повторения. Возвращает угол и для каждого повторения его k (0 наверху, 1 внизу)."""
    # подход начинается на прямых руках (на брусья запрыгивают / гриф снимают с прямыми руками)
    elbow, ks = [np.full(n_pause, top)], [np.zeros(n_pause)]
    for i, r in enumerate(reps):
        nxt = reps[i + 1].lock if i + 1 < len(reps) else top
        k = (1 - np.cos(np.linspace(0, 2 * np.pi, max(int(r.seconds * fps), 3)))) / 2
        base = np.where(np.arange(len(k)) < len(k) // 2, r.lock, nxt)  # откуда опускается / куда поднимается
        elbow += [np.full(n_pause, r.lock), base - (base - r.elbow) * k]
        ks += [np.zeros(n_pause), k]
    elbow.append(np.full(n_pause, top))
    ks.append(np.zeros(n_pause))
    return np.concatenate(elbow), ks


@dataclass
class DipsRep:
    elbow: float = 85.0  # угол локтя внизу
    drop: float = 0.1  # плечо ниже локтя внизу, в длинах плеча
    lock: float = 168.0  # угол локтя наверху ПЕРЕД этим повторением
    seconds: float = 2.2


DIPS_DEMO = {
    "shallow": DipsRep(elbow=125.0, drop=-0.5),
    "too_deep": DipsRep(elbow=65.0, drop=0.8),
    "no_lockout": DipsRep(lock=140.0),
    "fast": DipsRep(seconds=0.7),
}


def dips_synthetic(reps: list[DipsRep], fps: float = 30.0, pause_sec: float = 1.0, noise: float = 1.5,
                   seed: int = 0) -> dict[str, np.ndarray]:
    """Синтетические брусья: признаки, заданные формулой (для демо и тестов). Кисти на месте, тело ходит вверх-вниз."""
    rng = np.random.default_rng(seed)
    n_pause = int(pause_sec * fps)
    elbow, ks = _press_elbow(reps, fps, n_pause)
    k = np.concatenate(ks)
    drop = np.concatenate([np.full(len(c), -0.95) + (r.drop + 0.95) * c
                           for c, r in zip(ks, [DipsRep()] + [x for r in reps for x in (r, r)] + [DipsRep()])])
    feats = {
        "elbow_angle": smooth(elbow + rng.normal(0, noise, len(elbow)), 5),
        "shoulder_drop": smooth(drop + rng.normal(0, 0.02, len(drop)), 5),
        "sh_y": smooth(3.0 + 1.0 * k + rng.normal(0, 0.02, len(k)), 5),
        "wr_y": smooth(5.0 + rng.normal(0, 0.02, len(k)), 5),
    }
    feats["main_vel"] = np.gradient(feats["elbow_angle"]) * fps
    return feats


@dataclass
class BenchRep:
    elbow: float = 80.0  # угол локтя, когда гриф у груди
    hip: float = 0.0  # насколько таз поднимается выше плеч внизу, в длинах корпуса
    lock: float = 168.0  # угол локтя наверху ПЕРЕД этим повторением
    seconds: float = 2.2


BENCH_DEMO = {
    "no_chest": BenchRep(elbow=125.0),
    "butt_up": BenchRep(hip=0.4),
    "no_lockout": BenchRep(lock=140.0),
    "fast": BenchRep(seconds=0.7),
}


def bench_synthetic(reps: list[BenchRep], fps: float = 30.0, pause_sec: float = 1.0, noise: float = 1.5,
                    seed: int = 0) -> dict[str, np.ndarray]:
    """Синтетический жим лёжа: признаки, заданные формулой (для демо и тестов)."""
    rng = np.random.default_rng(seed)
    n_pause = int(pause_sec * fps)
    elbow, ks = _press_elbow(reps, fps, n_pause)
    hip = np.concatenate([r.hip * c for c, r in zip(ks, [BenchRep()] + [x for r in reps for x in (r, r)] + [BenchRep()])])
    feats = {
        "elbow_angle": smooth(elbow + rng.normal(0, noise, len(elbow)), 5),
        "hip_rise": smooth(hip + rng.normal(0, 0.02, len(hip)), 5),
    }
    feats["main_vel"] = np.gradient(feats["elbow_angle"]) * fps
    return feats


# для каждого упражнения: (ошибки демо -> параметры повторения, правильное повторение, генератор)
DEMOS = {
    "deadlift": (DEADLIFT_DEMO, DeadliftRep, deadlift_synthetic),
    "pushup": (PUSHUP_DEMO, PushupRep, pushup_synthetic),
    "pullup": (PULLUP_DEMO, PullupRep, pullup_synthetic),
    "dips": (DIPS_DEMO, DipsRep, dips_synthetic),
    "bench": (BENCH_DEMO, BenchRep, bench_synthetic),
}
DEMO_ERRORS = {key: list(d[0]) for key, d in DEMOS.items()}


def demo_session(spec: ExerciseSpec, errors: list[str], fps: float = 30.0) -> dict[str, np.ndarray]:
    """Демо: правильное повторение, потом по очереди выбранные ошибки, между ними — правильные."""
    bad, good, make = DEMOS[spec.key]
    plan = [good()]
    for e in errors:
        plan += [bad[e], good()]
    return make(plan, fps)
