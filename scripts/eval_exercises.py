"""Метрики HMM (неделя 7) и агента (неделя 1) для становой тяги, отжиманий и подтягиваний
на синтетических подходах с известным ответом — как scripts/eval_synthetic.py для приседа.

Запуск: python scripts/eval_exercises.py
Для каждого упражнения: много подходов со случайным числом повторений, случайными ошибками
случайной силы и разным шумом. Сравниваем счёт повторений HMM + агента с простым порогом
«главный угол ниже середины» и считаем precision / recall / F1 по каждой ошибке.
Результат: data/processed/exercises_report.json
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from formcoach.exercises import (EXERCISES, DeadliftRep, PullupRep, PushupRep, analyze_exercise,  # noqa: E402
                                 deadlift_synthetic, pullup_synthetic, pushup_synthetic)

FPS = 30.0


def U(rng, a, b) -> float:
    return float(rng.uniform(a, b))


def deadlift_rep(rng, kind):
    r = DeadliftRep(hip=U(rng, 85, 100), knee=U(rng, 115, 140), bar=U(rng, 0.15, 0.35), lean_back=U(rng, -5, 5),
                    seconds=U(rng, 2.0, 3.0))
    if kind == "bar_far":
        r.bar = U(rng, 0.7, 1.0)
    elif kind == "lean_back":
        r.lean_back = U(rng, 22, 35)
    elif kind == "squat_it":
        r.knee = U(rng, 75, 92)
    elif kind == "stiff_legs":
        r.knee = U(rng, 168, 176)
    elif kind == "fast":
        r.seconds = U(rng, 0.6, 0.85)
    return r


def pushup_rep(rng, kind):
    r = PushupRep(elbow=U(rng, 75, 95), sag=U(rng, 0, 8), pike=U(rng, 0, 8), seconds=U(rng, 1.5, 2.5))
    if kind == "shallow":
        r.elbow = U(rng, 115, 130)
    elif kind == "sag":
        r.sag = U(rng, 22, 35)
    elif kind == "pike":
        r.pike = U(rng, 28, 40)
    elif kind == "fast":
        r.seconds = U(rng, 0.45, 0.65)
    return r


def pullup_rep(rng, kind):
    r = PullupRep(elbow=U(rng, 35, 50), chin=U(rng, -0.3, -0.1), hang=U(rng, 165, 175), swing=U(rng, 4, 15),
                  seconds=U(rng, 1.8, 2.6))
    if kind == "chin_low":
        r.elbow, r.chin = U(rng, 65, 80), U(rng, 0.2, 0.5)
    elif kind == "no_hang":
        r.hang = U(rng, 130, 142)
    elif kind == "swing":
        r.swing = U(rng, 35, 45)
    elif kind == "fast":
        r.seconds = U(rng, 0.6, 0.8)
    return r


SETUP = {
    "deadlift": (deadlift_rep, deadlift_synthetic, ["bar_far", "lean_back", "squat_it", "stiff_legs", "fast"]),
    "pushup": (pushup_rep, pushup_synthetic, ["shallow", "sag", "pike", "fast"]),
    "pullup": (pullup_rep, pullup_synthetic, ["chin_low", "no_hang", "swing", "fast"]),
}


def threshold_count(x: np.ndarray, threshold: float) -> int:
    """Наивный счётчик: повторение = каждый раз, когда угол опустился ниже порога."""
    below = x < threshold
    return int(np.sum(below[1:] & ~below[:-1]))


def evaluate(key: str, n_sessions: int = 200, seed: int = 0) -> dict:
    spec = EXERCISES[key]
    make_rep, make_session, errors = SETUP[key]
    rng = np.random.default_rng(seed)
    count = {"hmm": [], "threshold": []}
    tp, fp, fn = (dict.fromkeys(errors, 0) for _ in range(3))
    times = []
    probs = [0.4] + [0.6 / len(errors)] * len(errors)
    for s in range(n_sessions):
        kinds = [str(rng.choice(["good", *errors], p=probs)) for _ in range(rng.integers(3, 9))]
        reps = [make_rep(rng, k) for k in kinds]
        feats = make_session(reps, fps=FPS, noise=U(rng, 1.0, 4.0), seed=s)
        t = time.perf_counter()
        a = analyze_exercise(spec, feats, FPS)
        times.append(time.perf_counter() - t)
        count["hmm"].append(len(a.reps) - len(reps))
        count["threshold"].append(threshold_count(feats[spec.main], (spec.top + spec.bottom) / 2) - len(reps))
        if len(a.reps) != len(reps):
            continue  # ошибки сравниваем, только когда повторения совпали один к одному
        for rep, kind in zip(a.reps, kinds):
            for e in errors:
                found, truth = e in rep.errors, e == kind
                tp[e] += found and truth
                fp[e] += found and not truth
                fn[e] += truth and not found

    def summary(diffs):
        d = np.array(diffs)
        return {"exact": float(np.mean(d == 0)), "mae": float(np.mean(np.abs(d)))}

    out_err = {}
    for e in errors:
        p = tp[e] / max(tp[e] + fp[e], 1)
        r = tp[e] / max(tp[e] + fn[e], 1)
        out_err[e] = {"precision": p, "recall": r, "f1": 2 * p * r / max(p + r, 1e-9), "support": tp[e] + fn[e]}
    return {"sessions": n_sessions, "count": {k: summary(v) for k, v in count.items()}, "errors": out_err,
            "analysis_ms_per_session": 1000 * float(np.mean(times))}


def main():
    report = {}
    for key in SETUP:
        r = report[key] = evaluate(key)
        print(f"{EXERCISES[key].text['ru']['name']}: {r['sessions']} подходов, шум 1–4°")
        print(f"  счёт повторений точно: HMM {r['count']['hmm']['exact']:.1%}, порог {r['count']['threshold']['exact']:.1%}")
        for e, m in r["errors"].items():
            print(f"  {e:10s} P {m['precision']:.3f}  R {m['recall']:.3f}  F1 {m['f1']:.3f}  (n={m['support']})")
        print(f"  анализ подхода: {r['analysis_ms_per_session']:.1f} мс")
    (ROOT / "data" / "processed" / "exercises_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
