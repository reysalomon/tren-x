"""Делает недостающие голосовые фразы живой камеры через ElevenLabs — тем же голосом, что уже записанные.

Запуск:  python scripts/make_voice.py            (сделать только то, чего ещё нет)
         python scripts/make_voice.py --list     (только показать, какие фразы будут сделаны)
         python scripts/make_voice.py --all --voice ID --model ID   (перезаписать ВСЕ фразы одним голосом)
         ... --out ПАПКА                         (писать не в web/audio, а в другую папку — чтобы сначала послушать)

Ключ ElevenLabs берётся из переменной ELEVENLABS_API_KEY или из файла ~/.elevenlabs_key (одна строка).
В репозиторий ключ не попадает.

Тексты фраз — из web/i18n.js (ключи say_...): казахские кладём в web/audio/kk/, русские в web/audio/ru/.
Голос и модель скрипт находит сам: смотрит историю генераций в ElevenLabs и берёт те, которыми были
записаны уже существующие фразы (например «Жақсы!»). Можно задать вручную: --voice ID --model ID.

Сейчас все фразы записаны голосом Callum (N2lVS1w4EtoT3dr4eOWO): первые 18 клипов делал другой человек,
его голос в аккаунте не нашёлся, поэтому всё перезаписано одним голосом — чтобы не звучало два диктора.
"""

import json
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
API = "https://api.elevenlabs.io/v1"


def phrases() -> dict[str, dict[str, str]]:
    """{"kk": {"swing": "Тербелме", ...}, "ru": {...}} из web/i18n.js."""
    text = (ROOT / "web" / "i18n.js").read_text(encoding="utf-8")
    cut = text.index("\n  ru: {")  # до этого места — казахский блок, после — русский
    find = lambda part: dict(re.findall(r'say_(\w+): "([^"]+)"', part))  # noqa: E731
    return {"kk": find(text[:cut]), "ru": find(text[cut:])}


def api_key() -> str:
    key = os.environ.get("ELEVENLABS_API_KEY", "").strip()
    file = Path.home() / ".elevenlabs_key"
    if not key and file.exists():
        key = file.read_text().strip()
    if not key:
        sys.exit("Нет ключа ElevenLabs: задай ELEVENLABS_API_KEY или положи ключ в ~/.elevenlabs_key")
    return key


def call(key: str, path: str, body: dict | None = None) -> bytes:
    req = urllib.request.Request(API + path, headers={"xi-api-key": key, "Content-Type": "application/json"},
                                 data=None if body is None else json.dumps(body).encode())
    try:
        return urllib.request.urlopen(req, timeout=120).read()
    except urllib.error.HTTPError as e:
        sys.exit(f"ElevenLabs ответил ошибкой {e.code} на {path}: {e.read().decode(errors='replace')[:300]}")


def voice_from_history(key: str, known: set[str]) -> tuple[str, str]:
    """Голос и модель, которыми записаны уже существующие фразы."""
    for item in json.loads(call(key, "/history?page_size=200"))["history"]:
        if item.get("text", "").strip() in known:
            return item["voice_id"], item["model_id"]
    sys.exit("В истории ElevenLabs не нашлось уже записанных фраз. Укажи голос и модель: --voice ID --model ID")


def arg(name: str) -> str | None:
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else None


def main():
    all_phrases = phrases()
    out = Path(arg("--out") or ROOT / "web" / "audio")
    todo = [(lang, name, text) for lang, items in all_phrases.items() for name, text in items.items()
            if "--all" in sys.argv or not (out / lang / f"{name}.mp3").exists()]
    if not todo:
        return print("Все фразы уже записаны.")
    print(f"Не хватает {len(todo)} фраз, {sum(len(t) for *_, t in todo)} символов:")
    for lang, name, text in todo:
        print(f"  {lang}/{name}.mp3  «{text}»")
    if "--list" in sys.argv:
        return

    key = api_key()
    voice, model = arg("--voice"), arg("--model")
    if not (voice and model):
        done = {t for items in all_phrases.values() for t in items.values()} - {t for *_, t in todo}
        found = voice_from_history(key, done)
        voice, model = voice or found[0], model or found[1]
    print(f"Голос {voice}, модель {model}")
    for lang, name, text in todo:
        audio = call(key, f"/text-to-speech/{voice}?output_format=mp3_44100_128", {"text": text, "model_id": model})
        (out / lang).mkdir(parents=True, exist_ok=True)
        (out / lang / f"{name}.mp3").write_bytes(audio)
        print(f"  готово {lang}/{name}.mp3 ({len(audio) // 1024} КБ)", flush=True)


if __name__ == "__main__":
    main()
