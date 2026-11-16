"""Скачивает открытые датасеты с реальными видео для scripts/eval_real.py (всего ~3 ГБ, в data/external/).

Запуск:  python scripts/fetch_real_datasets.py            (~3 ГБ)
         python scripts/fetch_real_datasets.py --train    (ещё ~7 ГБ: 94 новых видео подтягиваний из части train)

1. MyDeadlift — 191 видео становой тяги сбоку с метками техники (правильно / круглая спина / колени за носками).
   Mendeley Data, DOI 10.17632/w5prmmxyt9.1, лицензия CC BY 4.0. ~130 МБ.
2. RepCount-A — видео с YouTube с известным числом повторений (статья TransRAC, CVPR 2022, для учебных целей).
   Берём копию с Hugging Face (lmms-lab-eval/repcounta-lance), только части validation и test (~2.7 ГБ),
   и достаём из них приседания, отжимания и подтягивания (101 видео).
   С ключом --train дополнительно качается часть train, из неё берём только подтягивания: на этих видео
   проверяется правило "кисти выше плеч", придуманное по частям validation и test.
   Для чтения формата Lance нужен пакет:  uv pip install --python .venv/bin/python pylance

Сами видео в репозиторий не попадают (data/external/ в .gitignore).
"""

import csv
import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

EXT = Path(__file__).resolve().parent.parent / "data" / "external"
MENDELEY = "https://data.mendeley.com/public-api/datasets/w5prmmxyt9"
HF_API = "https://huggingface.co/api/datasets/lmms-lab-eval/repcounta-lance/tree/main?recursive=1"
HF_FILE = "https://huggingface.co/datasets/lmms-lab-eval/repcounta-lance/resolve/main/"
CLASSES = {"squat", "squant", "push_up", "pushups", "pull_up", "pullups", "bench_pressing", "benchpressing"}  # "squant" — опечатка в самом датасете
TRAIN_CLASSES = {"pull_up", "pullups", "bench_pressing", "benchpressing"}


def open_url(url: str):
    # без заголовка User-Agent сервер Mendeley отвечает 403
    return urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=60)


def download(url: str, path: Path, size: int | None = None):
    """Скачивает файл, если его ещё нет (или размер не совпал — значит, прошлый раз оборвалось)."""
    if path.exists() and (size is None or path.stat().st_size == size):
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    with open_url(url) as r, tmp.open("wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    tmp.rename(path)


def fetch_mydeadlift():
    out = EXT / "mydeadlift"
    folders = [("root", "")] + [(f["id"], f["name"].replace(" ", "_")) for f in json.load(open_url(MENDELEY + "/folders/1"))]
    for folder_id, name in folders:
        files = json.load(open_url(f"{MENDELEY}/files?folder_id={folder_id}&version=1"))
        for f in files:
            download(f["content_details"]["download_url"], out / name / f["filename"], f["size"])
        print(f"MyDeadlift/{name or '.'}: {len(files)} файлов", flush=True)


def fetch_repcount(train: bool = False):
    out = EXT / "repcount"
    splits = ("validation", "test", "train") if train else ("validation", "test")
    for f in json.load(open_url(HF_API)):
        p = f["path"]
        if f["type"] == "file" and p.startswith(tuple(f"data/{s}.lance" for s in splits)):
            print("RepCount:", p, f"{f['size'] / 1e6:.0f} МБ", flush=True)
            download(HF_FILE + urllib.parse.quote(p), out / p, f["size"])
    try:
        import lance
    except ImportError:
        sys.exit("Файлы скачаны, но для распаковки нужен пакет pylance (см. начало этого файла).")
    rows = []
    (out / "videos").mkdir(exist_ok=True)
    for split in splits:
        ds = lance.dataset(str(out / "data" / f"{split}.lance"))
        meta = ds.to_table(columns=["video_id", "action_type", "count", "video_ext"]).to_pylist()
        idx = [i for i, m in enumerate(meta) if m["action_type"] in (TRAIN_CLASSES if split == "train" else CLASSES)]
        for i, blob in zip(idx, ds.take_blobs("video_blob", indices=idx)):
            name = meta[i]["video_id"] + "." + str(meta[i]["video_ext"]).lstrip(".")
            with blob as b:
                (out / "videos" / name).write_bytes(b.read())
            rows.append([name, meta[i]["action_type"], meta[i]["count"], split])
    with (out / "labels.csv").open("w", newline="") as f:
        csv.writer(f).writerows([["file", "action_type", "count", "split"], *rows])
    print(f"RepCount: {len(rows)} видео, метки в {out / 'labels.csv'}")


if __name__ == "__main__":
    fetch_mydeadlift()
    fetch_repcount(train="--train" in sys.argv)
