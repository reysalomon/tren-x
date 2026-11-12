"""Проверка TREN-X на ЧУЖИХ реальных видео из открытых датасетов (не на синтетике и не на своих видео).

Запуск:  python scripts/eval_real.py            (сначала scripts/fetch_real_datasets.py — скачать видео)

Датасеты:
  RepCount-A (CVPR 2022, видео с YouTube)  — приседания, отжимания, подтягивания; известно число повторений.
      Проверяем неделю 7: счёт повторений HMM + агента против простого порога.
  MyDeadlift (Mendeley Data, CC BY 4.0)    — 191 видео становой тяги сбоку, по одному повторению,
      метки: правильно / круглая спина / колени за носками.
      Проверяем счёт (должно быть 1), ложные замечания агента (неделя 1) на правильных повторениях
      и сравниваем наши углы с углами авторов датасета.

Точки тела (MediaPipe) считаются долго, поэтому сохраняются в data/processed/real_poses/.
Результат: data/processed/real_report.json
"""

import csv
import json
import sys
from dataclasses import replace
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from formcoach import pose as P  # noqa: E402
from formcoach.exercises import EXERCISES, analyze_exercise, default_hmm, exercise_features  # noqa: E402
from formcoach.features import squat_features  # noqa: E402
from formcoach.pipeline import analyze_squat  # noqa: E402

EXT = ROOT / "data" / "external"
CACHE = ROOT / "data" / "processed" / "real_poses"
MIN_DETECTED = 0.5  # если человек найден меньше чем на половине кадров — видео не подходит для анализа

# названия классов RepCount (в датасете есть опечатки) -> наше упражнение
REPCOUNT_CLASSES = {"squat": "squat", "squant": "squat", "push_up": "pushup", "pushups": "pushup",
                    "pull_up": "pullup", "pullups": "pullup"}


def cached_pose(job: tuple[str, str]) -> str:
    """Считает точки тела для одного видео и кладёт в кэш. Возвращает текст ошибки или ''."""
    video, out = job
    if Path(out).exists():
        return ""
    try:
        seq = P.extract_pose(video)
    except Exception as e:  # битое видео не должно ронять всю проверку
        return f"{video}: {e}"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, landmarks=seq.landmarks.astype(np.float32), fps=seq.fps)
    return ""


def load_pose(path: Path) -> P.PoseSequence:
    d = np.load(path)
    return P.PoseSequence(landmarks=d["landmarks"].astype(float), fps=float(d["fps"]))


def threshold_count(x: np.ndarray, threshold: float) -> int:
    """Наивный счётчик для сравнения: повторение = угол опустился ниже порога."""
    below = x < threshold
    return int(np.sum(below[1:] & ~below[:-1]))


def analyze(key: str, seq: P.PoseSequence, pose_rule: bool = True):
    """Тот же путь, что на сайте: признаки -> HMM -> агент. Возвращает (анализ, главный угол, середина для порога).
    pose_rule=False выключает проверку позы (у подтягиваний — "кисти выше плеч"), чтобы измерить её пользу."""
    if key == "squat":
        feats = squat_features(seq)
        return analyze_squat(feats, seq.fps), feats["knee_angle"], 125.0  # середина между 165 и 85
    spec = EXERCISES[key] if pose_rule else replace(EXERCISES[key], pose_ok=None)
    feats = exercise_features(spec, seq)
    h = default_hmm(spec)
    return analyze_exercise(spec, feats, seq.fps), feats[spec.main], (h["top_angle"] + h["bottom_angle"]) / 2


def count_summary(diffs: list[int]) -> dict:
    d = np.array(diffs)
    return {"exact": float(np.mean(d == 0)), "off_by_one": float(np.mean(np.abs(d) <= 1)), "mae": float(np.mean(np.abs(d)))}


# ---------------- RepCount-A ----------------


def repcount_jobs() -> list[dict]:
    labels = EXT / "repcount" / "labels.csv"
    if not labels.exists():
        return []
    rows = []
    for r in csv.DictReader(labels.open()):
        if r["action_type"] in REPCOUNT_CLASSES and r["count"].isdigit():
            rows.append({"video": str(EXT / "repcount" / "videos" / r["file"]), "count": int(r["count"]),
                         "split": r["split"],
                         "exercise": REPCOUNT_CLASSES[r["action_type"]],
                         "pose": str(CACHE / "repcount" / (Path(r["file"]).stem + ".npz"))})
    return rows


def eval_repcount(rows: list[dict], keys=("squat", "pushup", "pullup"), pose_rule: bool = True) -> dict:
    out = {}
    for key in keys:
        hmm, thr, skipped, per_video = [], [], 0, []
        for r in (r for r in rows if r["exercise"] == key):
            if not Path(r["pose"]).exists():
                skipped += 1
                continue
            seq = load_pose(Path(r["pose"]))
            try:
                if seq.detected_ratio < MIN_DETECTED:
                    raise ValueError("человек найден меньше чем на половине кадров")
                a, main, mid = analyze(key, seq, pose_rule)
            except ValueError:
                skipped += 1
                continue
            hmm.append(len(a.reps) - r["count"])
            thr.append(threshold_count(main, mid) - r["count"])
            per_video.append({"video": Path(r["video"]).name, "true": r["count"], "hmm": len(a.reps),
                              "threshold": thr[-1] + r["count"]})
        if hmm:
            out[key] = {"videos": len(hmm), "skipped": skipped, "hmm": count_summary(hmm),
                        "threshold": count_summary(thr), "per_video": per_video}
    return out


# ---------------- MyDeadlift ----------------

DEADLIFT_LABELS = {"Gerakan_Benar": "correct", "Punggung_Bungkuk": "rounded_back", "Lutut_Lebih_Jari_Kaki": "knees_forward"}


def deadlift_jobs() -> list[dict]:
    rows = []
    for folder, label in DEADLIFT_LABELS.items():
        for v in sorted((EXT / "mydeadlift" / folder).glob("*.mp4")):
            rows.append({"video": str(v), "label": label, "pose": str(CACHE / "mydeadlift" / (v.stem + ".npz"))})
    return rows


def author_table() -> tuple[dict[str, dict[int, tuple[float, float]]], dict[str, str]]:
    """Таблица авторов датасета: углы колена и таза (видео -> кадр -> (колено, таз))
    и деление по людям (видео -> "training" / "testing"; в тестовой части — два других человека)."""
    path = EXT / "mydeadlift" / "Extracted_keypoints.csv"
    angles, split = defaultdict(dict), {}
    if path.exists():
        for r in csv.DictReader(path.open(), delimiter=";"):
            num = lambda s: float(s.replace(",", "."))  # noqa: E731 — в файле десятичная запятая
            angles[r["Video"]][int(r["Frame"])] = (num(r["Sudut_Lutut"]), num(r["Sudut_Pinggul"]))
            split[r["Video"]] = r["Split"]
    return angles, split


def prf(pred: list[bool], truth: list[bool]) -> dict:
    tp = sum(p and t for p, t in zip(pred, truth))
    p, r = tp / max(sum(pred), 1), tp / max(sum(truth), 1)
    return {"precision": p, "recall": r, "f1": 2 * p * r / max(p + r, 1e-9), "videos": len(pred)}


def eval_deadlift(rows: list[dict]) -> dict:
    spec = EXERCISES["deadlift"]
    theirs, split = author_table()
    knees = defaultdict(lambda: ([], []))  # часть датасета -> (агент сказал "присед", метка "колени вперёд")
    by_label = defaultdict(lambda: {"videos": 0, "count_diffs": [], "clean": 0, "errors": Counter()})
    knee_diff, hip_diff, skipped = [], [], 0
    for r in rows:
        if not Path(r["pose"]).exists():
            skipped += 1
            continue
        seq = load_pose(Path(r["pose"]))
        try:
            feats = exercise_features(spec, seq)
            a = analyze_exercise(spec, feats, seq.fps)
        except ValueError:
            skipped += 1
            continue
        b = by_label[r["label"]]
        b["videos"] += 1
        b["count_diffs"].append(len(a.reps) - 1)
        errors = {e for rep in a.reps for e in rep.errors}
        b["clean"] += bool(a.reps) and not errors
        b["errors"].update(errors)
        part = knees[split.get(Path(r["video"]).name, "training")]
        part[0].append("squat_it" in errors)
        part[1].append(r["label"] == "knees_forward")
        for frame, (knee, hip) in theirs.get(Path(r["video"]).name, {}).items():
            if frame - 1 < len(feats["knee_angle"]):  # у авторов кадры с 1
                knee_diff.append(abs(feats["knee_angle"][frame - 1] - knee))
                hip_diff.append(abs(feats["hip_angle"][frame - 1] - hip))
    out = {"skipped": skipped, "labels": {}}
    for label, b in by_label.items():
        out["labels"][label] = {"videos": b["videos"], "count": count_summary(b["count_diffs"]),
                                "no_remarks": b["clean"] / b["videos"],
                                "errors": {e: n / b["videos"] for e, n in b["errors"].most_common()}}
    # Ошибку датасета "колени за носками" наш агент видит как "слишком согнутые колени" (squat_it).
    # Порог угла колена подобран на обучающей части, поэтому честная оценка — на тестовой (другие люди).
    out["knees_forward_as_squat_it"] = {name: prf(*knees[name]) for name in ("training", "testing") if name in knees}
    if knee_diff:
        out["angles_vs_authors"] = {"frames": len(knee_diff), "knee_mae": float(np.nanmean(knee_diff)),
                                    "hip_mae": float(np.nanmean(hip_diff))}
    return out


def main():
    rc, dl = repcount_jobs(), deadlift_jobs()
    if not rc and not dl:
        sys.exit("Нет видео. Сначала: python scripts/fetch_real_datasets.py")
    jobs = [(r["video"], r["pose"]) for r in rc + dl if not Path(r["pose"]).exists()]
    if jobs:
        print(f"MediaPipe: {len(jobs)} видео (один раз, потом из кэша)...")
        with ProcessPoolExecutor(max_workers=4) as pool:
            for i, err in enumerate(pool.map(cached_pose, jobs), 1):
                if err:
                    print("  пропуск:", err)
                if i % 20 == 0:
                    print(f"  {i}/{len(jobs)}", flush=True)

    report = {}
    if rc:
        # Части validation и test — на них мы смотрели, когда искали причину ошибок в подтягиваниях.
        # Часть train (только подтягивания) мы до этого не видели: это честная проверка правила "кисти выше плеч".
        dev, held = [r for r in rc if r["split"] != "train"], [r for r in rc if r["split"] == "train"]
        report["repcount"] = eval_repcount(dev)
        if held:
            report["repcount"]["pullup_new_videos"] = eval_repcount(held, ("pullup",))["pullup"]
            report["repcount"]["pullup_new_videos_no_rule"] = eval_repcount(held, ("pullup",), pose_rule=False)["pullup"]
        report["repcount"]["pullup_no_rule"] = eval_repcount(dev, ("pullup",), pose_rule=False)["pullup"]
        print("\nRepCount-A — счёт повторений (точно / ±1 / средняя ошибка):")
        for key, r in report["repcount"].items():
            h, t = r["hmm"], r["threshold"]
            print(f"  {key:25s} видео {r['videos']:3d} (пропущено {r['skipped']})  "
                  f"HMM {h['exact']:.0%} / {h['off_by_one']:.0%} / {h['mae']:.2f}   "
                  f"порог {t['exact']:.0%} / {t['off_by_one']:.0%} / {t['mae']:.2f}")
    if dl:
        report["mydeadlift"] = d = eval_deadlift(dl)
        print("\nMyDeadlift — одно повторение на видео:")
        for label, r in d["labels"].items():
            print(f"  {label:14s} видео {r['videos']:3d}  счёт точно {r['count']['exact']:.0%}  "
                  f"без замечаний {r['no_remarks']:.0%}  ошибки {r['errors']}")
        for name, m in d["knees_forward_as_squat_it"].items():
            print(f"  «колени вперёд» ({name}, {m['videos']} видео): P {m['precision']:.2f}  R {m['recall']:.2f}  F1 {m['f1']:.2f}")
        if "angles_vs_authors" in d:
            m = d["angles_vs_authors"]
            print(f"  наши углы против углов авторов: колено ±{m['knee_mae']:.1f}°, таз ±{m['hip_mae']:.1f}° ({m['frames']} кадров)")
    (ROOT / "data" / "processed" / "real_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
