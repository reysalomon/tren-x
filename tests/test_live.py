"""Живой режим (камера): потоковый анализ должен давать то же, что анализ целого видео."""

from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

import server
from formcoach.features import squat_features
from formcoach.hmm import squat_hmm
from formcoach.live import LiveSession
from formcoach.pipeline import analyze_squat
from formcoach.pose import PoseSequence

OWN_LANDMARKS = Path(__file__).resolve().parent.parent / "data" / "labels" / "landmarks" / "f691fd75ae576ae48ec5e8ae25afeb7e.npy"
needs_video = pytest.mark.skipif(not OWN_LANDMARKS.exists(), reason="нет размеченного видео")


def test_filter_step_matches_forward_algorithm():
    """Сумма шаговых нормировок фильтрации = log P(всё видео) из прямого алгоритма."""
    hmm = squat_hmm()
    rng = np.random.default_rng(0)
    X = np.column_stack([rng.uniform(80, 170, 50), rng.normal(0, 60, 50)])
    alpha, total = None, 0.0
    for x in X:
        alpha, step = hmm.filter_step(alpha, x)
        total += step
        assert np.isclose(np.exp(alpha).sum(), 1.0)  # это вероятности фаз на текущем кадре
    assert total == pytest.approx(hmm.log_likelihood(X))


@needs_video
@pytest.mark.parametrize("fps", [30, 15, 10, 5])
def test_live_same_reps_as_video(fps):
    """Подаём точки настоящего видео по одному кадру, как с камеры (и реже — как на слабом телефоне)."""
    lm = np.load(OWN_LANDMARKS)
    video = analyze_squat(squat_features(PoseSequence(lm, 30.0)), 30.0)
    live = LiveSession()
    for i in range(0, len(lm), 30 // fps):
        live.push(i / 30.0, None if np.isnan(lm[i, 0, 0]) else lm[i])
    assert [r.errors for r in live.agent.reps] == [r.errors for r in video.reps]


def test_live_without_person_counts_nothing():
    s = LiveSession()
    for i in range(30):
        out = s.push(i / 30.0, None)
    assert out["person"] is False and out["reps"] == 0
    with pytest.raises(ValueError):
        LiveSession().summary()


@needs_video
def test_websocket_live_protocol():
    lm = np.load(OWN_LANDMARKS)
    client = TestClient(server.app)
    with client.websocket_connect("/ws/live") as ws:
        ws.send_json({"type": "frame", "t": 0, "lm": None})  # кадр до start — ошибка, но соединение живо
        assert ws.receive_json()["type"] == "error"
        ws.send_json({"type": "start", "lang": "kk", "settings": {"max_torso_lean": 30}})
        assert ws.receive_json() == {"type": "ready"}
        codes = []
        for i in range(len(lm)):
            ws.send_json({"type": "frame", "t": i / 30, "lm": None if np.isnan(lm[i, 0, 0]) else lm[i].tolist()})
            state = ws.receive_json()
            codes += [e["code"] for e in state["events"]]
        assert state["reps"] == 4 and "rep" in codes and "lean" in codes
        ws.send_json({"type": "frame", "t": 99, "lm": [1, 2, 3]})  # неверные точки
        assert ws.receive_json()["detail"].startswith("Камерамен")
        ws.send_json({"type": "stop"})
        result = ws.receive_json()
        assert result["type"] == "result" and result["data"]["summary"]["reps"] == 4
        assert result["data"]["reps"][0]["errors"][0]["text"]["kk"]
        # видео не присылали, но в карточке повторения — скелет в нижней точке
        assert all(r["image"].startswith("data:image/jpeg;base64,") for r in result["data"]["reps"])


def test_websocket_rejects_bad_settings():
    client = TestClient(server.app)
    with client.websocket_connect("/ws/live") as ws:
        ws.send_json({"type": "start", "lang": "ru", "settings": {"stay_prob": 5}})
        msg = ws.receive_json()
        assert msg["type"] == "error" and "stay_prob" in msg["detail"]
        ws.send_text("не json")
        assert ws.receive_json()["type"] == "error"


def test_phase_does_not_get_stuck_in_descent():
    """Человек встал, не дойдя до низа: из СПУСКА в ВЕРХ напрямую нельзя — идём через ПОДЪЁМ, а не застреваем."""
    from formcoach.hmm import DOWN, TOP, UP
    s = LiveSession()
    s.phases = [DOWN]
    s.log_alpha = np.log(np.array([0.9, 0.05, 0.01, 0.04]))  # фильтрация уверена: человек стоит (ВЕРХ)
    assert s._next_phase() == UP
    s.phases.append(UP)
    assert s._next_phase() == TOP
