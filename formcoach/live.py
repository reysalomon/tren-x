"""Живой режим: камера телефона -> тренер в реальном времени.

Телефон сам находит 33 точки тела (MediaPipe в браузере) и присылает их серверу по одному кадру.
Видео при этом никуда не уходит. Здесь — та же цепочка, что в pipeline.py, но потоковая:

  точки кадра -> признаки (скользящее среднее только по прошлым кадрам)
              -> HMM: прямой алгоритм-фильтрация (неделя 7), фаза меняется только по разрешённым переходам
              -> агент-тренер: тот же SquatCoachAgent.step(), что и для видео (неделя 1)

Отличие от видео: будущих кадров нет. Поэтому вместо Витерби (смотрит на всё видео) — фильтрация,
а сглаживание и скорость считаются только по уже пришедшим кадрам (небольшая задержка ~0.1 с).
"""

import numpy as np

from . import pose as P
from .agent import Feedback, Percept, SquatCoachAgent, SquatRules
from .features import FOOT_MIN_VISIBILITY, joint_angle, lean_from_vertical
from .hmm import squat_hmm
from .pipeline import Analysis

FEATURES = ["knee_angle", "hip_angle", "torso_lean", "thigh_angle", "knee_vel", "heel_lift", "hip_y", "ankle_lx", "ankle_rx"]
MIN_VISIBILITY = 0.5  # бедро/колено/лодыжка видны хуже — человека на кадре "нет"


def hmm_for_step(hmm_params: dict, dt_frame: float):
    """HMM для одного шага живой камеры с учётом реального времени между кадрами.

    stay_prob задан для 30 кадр/с. Если кадры приходят реже, за один шаг проходит больше времени,
    и вероятность остаться в фазе должна быть меньше: stay^(30 * dt). Иначе модель "не успевает" за движением.
    """
    stay = hmm_params.get("stay_prob", 0.9) ** float(np.clip(30 * dt_frame, 0.5, 10))
    return squat_hmm(**{**hmm_params, "stay_prob": float(np.clip(stay, 0.05, 0.99))})


def next_phase(allowed: np.ndarray, log_alpha: np.ndarray, prev: int | None) -> int:
    """Фаза на этом кадре: самая вероятная по фильтрации, но меняется только по разрешённым переходам.

    Без ограничения шум мог бы "перепрыгнуть" из НИЗА сразу в ВЕРХ и посчитать лишнее повторение.
    Но если самая вероятная фаза недостижима за один шаг (человек встал, а модель ещё в СПУСКЕ),
    идём к ней через промежуточную фазу (СПУСК -> ПОДЪЁМ -> ВЕРХ), а не застреваем на месте.
    """
    best = int(np.argmax(log_alpha))
    if prev is None or allowed[prev, best]:
        return best
    bridges = [j for j in range(len(log_alpha)) if j != prev and allowed[prev, j] and allowed[j, best]]
    if bridges:
        return max(bridges, key=lambda j: log_alpha[j])
    return int(np.argmax(np.where(allowed[prev], log_alpha, -np.inf)))



class LiveSession:
    """Один подход перед камерой. push() вызывается на каждый кадр, summary() — в конце."""

    def __init__(self, hmm_params: dict | None = None, rules: SquatRules | None = None, window: int = 5):
        self.hmm_params = dict(hmm_params or {})
        self.hmm = squat_hmm(**self.hmm_params)
        self.allowed = self.hmm.trans > 0  # какие переходы между фазами вообще возможны
        self.agent = SquatCoachAgent(rules, fps=30.0)
        self.window = window
        self.times: list[float] = []
        self.landmarks: list[np.ndarray] = []
        self.raw: dict[str, list[float]] = {k: [] for k in ["knee", "hip", "lean", "thigh", "heel", "foot_vis", "hip_y", "alx", "arx", "shin"]}
        self.features: dict[str, list[float]] = {k: [] for k in FEATURES}
        self.phases: list[int] = []
        self.feedback: list[Feedback] = []
        self.log_alpha: np.ndarray | None = None
        self.log_likelihood = 0.0
        self.vis_sum = np.zeros(2)  # накопленная видимость левой и правой ноги — берём сторону, что видна лучше

    # ---------- признаки одного кадра ----------

    def _raw_values(self, pts: np.ndarray | None) -> dict[str, float]:
        nan = dict.fromkeys(self.raw, np.nan)
        if pts is None or np.isnan(pts[:, :2]).any():
            return nan
        self.vis_sum += [pts[[P.L_HIP, P.L_KNEE, P.L_ANKLE], 3].mean(), pts[[P.R_HIP, P.R_KNEE, P.R_ANKLE], 3].mean()]
        if self.vis_sum[0] >= self.vis_sum[1]:
            sh, hip, knee, ankle, heel, foot = P.L_SHOULDER, P.L_HIP, P.L_KNEE, P.L_ANKLE, P.L_HEEL, P.L_FOOT
        else:
            sh, hip, knee, ankle, heel, foot = P.R_SHOULDER, P.R_HIP, P.R_KNEE, P.R_ANKLE, P.R_HEEL, P.R_FOOT
        if pts[[hip, knee], 3].min() < MIN_VISIBILITY:
            return nan
        xy = pts[:, :2]
        thigh = xy[knee] - xy[hip]
        shin = float(np.linalg.norm(xy[knee] - xy[ankle]))
        return {
            "knee": float(joint_angle(xy[hip], xy[knee], xy[ankle])),
            "hip": float(joint_angle(xy[sh], xy[hip], xy[knee])),
            "lean": float(lean_from_vertical(xy[sh], xy[hip])),
            "thigh": float(np.degrees(np.arctan2(thigh[1], abs(thigh[0])))),
            "heel": float((xy[foot, 1] - xy[heel, 1]) / max(shin, 1e-6)),
            "foot_vis": float(min(pts[heel, 3], pts[foot, 3])),
            "hip_y": float((xy[P.L_HIP, 1] + xy[P.R_HIP, 1]) / 2),
            "alx": float(xy[P.L_ANKLE, 0]),
            "arx": float(xy[P.R_ANKLE, 0]),
            "shin": shin,
        }

    def _avg(self, key: str) -> float:
        """Скользящее среднее по последним кадрам (только прошлое — будущего у камеры нет).
        Окно — по времени, как 5 кадров при 30 кадр/с (~0.17 с): на медленном телефоне кадров в окне меньше."""
        n = max(1, round(self.window * self._fps() / 30))
        vals = np.array(self.raw[key][-n:])
        vals = vals[~np.isnan(vals)]
        return float(vals.mean()) if len(vals) else np.nan

    def _fps(self) -> float:
        """Телефон присылает кадры неравномерно, поэтому fps оцениваем по последним ~2 секундам."""
        t = self.times[-60:]
        return (len(t) - 1) / (t[-1] - t[0]) if len(t) > 1 and t[-1] > t[0] else 30.0

    # ---------- главный шаг ----------

    def push(self, t: float, points) -> dict:
        """Новый кадр: t — время в секундах, points — (33, 4): x, y в пикселях, z, visibility; None — человека нет."""
        pts = None if points is None else np.asarray(points, dtype=float).reshape(33, 4)
        self.times.append(float(t))
        self.landmarks.append(np.full((33, 4), np.nan) if pts is None else pts)
        raw = self._raw_values(pts)
        for k, v in raw.items():
            self.raw[k].append(v)

        person = not np.isnan(raw["knee"])
        knee = self._avg("knee") if person else np.nan
        shins = np.array(self.raw["shin"])
        shin = float(np.median(shins[~np.isnan(shins)])) if person else np.nan
        # скорость колена: разница сглаженного угла за 2 кадра / прошедшее время (град/с)
        prev = self.features["knee_angle"][-2] if len(self.features["knee_angle"]) >= 2 else np.nan
        dt = self.times[-1] - self.times[-3] if len(self.times) >= 3 else 0.0
        vel = (knee - prev) / dt if dt > 0 and not np.isnan(prev) and person else 0.0
        heel = np.nan
        if person and self._avg("foot_vis") >= FOOT_MIN_VISIBILITY:
            # подъём пятки относительно обычного положения (медиана: большую часть времени пятки на полу)
            heel = self._avg("heel") - float(np.nanmedian(self.raw["heel"]))
        values = {
            "knee_angle": knee, "hip_angle": self._avg("hip"), "torso_lean": self._avg("lean"),
            "thigh_angle": self._avg("thigh"), "knee_vel": vel, "heel_lift": heel,
            "hip_y": self._avg("hip_y") / shin, "ankle_lx": self._avg("alx") / shin, "ankle_rx": self._avg("arx") / shin,
        }
        for k, v in values.items():
            self.features[k].append(float(v))

        events: list[Feedback] = []
        if not person:
            self.phases.append(self.phases[-1] if self.phases else 0)  # человека не видно — фазу не меняем
        else:
            dt_frame = self.times[-1] - self.times[-2] if len(self.times) > 1 else 1 / 30
            self.hmm = hmm_for_step(self.hmm_params, dt_frame)
            self.log_alpha, step_ll = self.hmm.filter_step(self.log_alpha, [knee, vel])
            self.log_likelihood += step_ll
            phase = self._next_phase()
            self.phases.append(phase)
            self.agent.fps = self._fps()
            frame = len(self.times) - 1
            events = self.agent.step(Percept(frame, phase, knee, values["torso_lean"], heel, values["thigh_angle"],
                                             values["hip_y"], values["ankle_lx"], values["ankle_rx"]))
            self.feedback += events
        return {
            "person": person,
            "phase": self.phases[-1],
            "knee": None if np.isnan(knee) else round(float(knee), 1),
            "reps": len(self.agent.reps),
            "good": sum(r.good for r in self.agent.reps),
            "events": events,
        }

    def _next_phase(self) -> int:
        return next_phase(self.allowed, self.log_alpha, self.phases[-1] if self.phases else None)

    def summary(self, classifier=None) -> Analysis:
        """Итог подхода в том же виде, что и анализ видео, — сайт показывает его на экране «Нәтиже»."""
        if len(self.times) < 2:
            raise ValueError("Слишком короткий подход.")
        span = self.times[-1] - self.times[0]
        fps = (len(self.times) - 1) / span if span > 0 else 30.0
        landmarks = np.array(self.landmarks)
        ml = [None] * len(self.agent.reps)
        if classifier is not None:
            from .classifier import classify_rep
            ml = [classify_rep(classifier, landmarks, r.bottom_frame) for r in self.agent.reps]
        features = {k: np.array(v) for k, v in self.features.items()}
        return Analysis(features, np.array(self.phases), self.log_likelihood, self.agent.reps, self.feedback, fps, ml)
