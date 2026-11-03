"""Становая тяга (formcoach/exercises.py): признаки, агент, живой режим и сервер."""

import numpy as np
import pytest
from fastapi.testclient import TestClient

import server
from formcoach import pose as P
from formcoach.exercises import (DEADLIFT, DEADLIFT_DEMO, ExerciseLiveSession, analyze_exercise, deadlift_values,
                                 demo_session, exercise_features)

client = TestClient(server.app)


def skeleton(lean: float, knee: float = 170.0, facing: int = 1, bar: float = 0.0) -> np.ndarray:
    """Человек сбоку (33 точки, пиксели). lean — наклон корпуса вперёд от вертикали (<0 — назад),
    knee — угол колена, facing — смотрит вправо (1) или влево (-1), bar — кисти дальше от ног (в голенях)."""
    pts = np.zeros((33, 4))
    pts[:, 3] = 0.95
    r = np.radians
    tilt = (180 - knee) / 2  # голень и бедро наклонены одинаково: колено уходит вперёд
    ankle = np.array([300.0, 600.0])
    knee_pt = ankle + 100 * np.array([np.sin(r(tilt)) * facing, -np.cos(r(tilt))])
    hip = knee_pt + 100 * np.array([-np.sin(r(tilt)) * facing, -np.cos(r(tilt))])
    sh = hip + 120 * np.array([np.sin(r(lean)) * facing, -np.cos(r(lean))])
    wrist = np.array([ankle[0] + (20 + bar * 100) * facing, sh[1] + 110])
    for side in ("L", "R"):
        g = lambda name: getattr(P, f"{side}_{name}")  # noqa: E731
        pts[g("SHOULDER"), :2], pts[g("HIP"), :2], pts[g("KNEE"), :2], pts[g("ANKLE"), :2] = sh, hip, knee_pt, ankle
        pts[g("ELBOW"), :2] = (sh + wrist) / 2
        pts[g("WRIST"), :2] = wrist
        pts[g("HEEL"), :2] = ankle + [-15 * facing, 5]
        pts[g("FOOT"), :2] = ankle + [40 * facing, 5]
    return pts


def session(reps: int = 3, facing: int = 1, fps: float = 30.0, bad: dict | None = None) -> np.ndarray:
    """Подход: стоит 1 с, потом reps наклонов по 2.5 с; bad — ошибка во втором повторении:
    {"bar": удаление штанги}, {"knee": угол колена внизу}, {"lean_back": наклон назад, когда встал}."""
    frames = [skeleton(0, facing=facing)] * int(fps)
    for i in range(reps):
        extra = bad if (bad and i == 1) else {}
        for k in (1 - np.cos(np.linspace(0, 2 * np.pi, int(2.5 * fps)))) / 2:
            knee = 170 - (170 - extra.get("knee", 130)) * k
            frames.append(skeleton(75 * k, knee, facing, extra.get("bar", 0.1) * k))
        frames += [skeleton(-extra.get("lean_back", 0), facing=facing)] * int(0.5 * fps)
        frames += [skeleton(0, facing=facing)] * int(fps)
    return np.array(frames)


def video(lm: np.ndarray, fps: float = 30.0):
    return analyze_exercise(DEADLIFT, exercise_features(DEADLIFT, P.PoseSequence(lm, fps)), fps)


def test_values_angles_and_direction():
    for facing in (1, -1):
        v = deadlift_values(skeleton(60, 130, facing), "L", facing)
        assert v["knee_angle"] == pytest.approx(130, abs=0.5)
        assert v["hip_angle"] == pytest.approx(180 - 25 - 60, abs=0.5)  # 180 - наклон бедра - наклон корпуса
        assert v["lean_back"] == 0  # наклон вперёд — не "назад"
        assert deadlift_values(skeleton(-20, 175, facing), "L", facing)["lean_back"] == pytest.approx(20, abs=0.5)


@pytest.mark.parametrize("facing", [1, -1])
def test_video_counts_reps_both_directions(facing):
    a = video(session(3, facing))
    assert len(a.reps) == 3
    assert all(r.good for r in a.reps), [r.errors for r in a.reps]


@pytest.mark.parametrize("bad, code", [({"bar": 0.9}, "bar_far"), ({"knee": 80}, "squat_it"),
                                       ({"knee": 172}, "stiff_legs"), ({"lean_back": 25}, "lean_back")])
def test_video_finds_errors_on_skeleton(bad, code):
    a = video(session(3, bad=bad))
    assert len(a.reps) == 3
    assert code in a.reps[1].errors and a.reps[0].good, [r.errors for r in a.reps]


@pytest.mark.parametrize("fps", [30, 15, 10])
def test_live_same_as_video(fps):
    lm = session(3, bad={"bar": 0.9})
    expected = [r.errors for r in video(lm).reps]
    live = ExerciseLiveSession(DEADLIFT)
    for i in range(0, len(lm), 30 // fps):
        live.push(i / 30.0, lm[i])
    assert [r.errors for r in live.agent.reps] == expected
    assert len(live.summary().reps) == 3


def test_live_without_person():
    s = ExerciseLiveSession(DEADLIFT)
    for i in range(10):
        out = s.push(i / 30, None)
    assert out["person"] is False and out["reps"] == 0
    with pytest.raises(ValueError):
        s.summary()


@pytest.mark.parametrize("error", list(DEADLIFT_DEMO))
def test_demo_each_error(error):
    a = analyze_exercise(DEADLIFT, demo_session(DEADLIFT, [error]), 30.0)
    assert [r.errors for r in a.reps] == [[], [error], []]


def test_server_demo_and_info():
    r = client.get("/api/demo", params={"exercise": "deadlift", "errors": "bar_far,fast", "lang": "kk"})
    assert r.status_code == 200
    data = r.json()
    assert data["exercise"] == "deadlift" and data["summary"]["reps"] == 5
    assert [e["key"] for e in data["reps"][1]["errors"]] == ["bar_far"]
    assert data["reps"][1]["metrics"][3]["over"] is True  # штанга — превышен порог
    assert data["phase_names"]["kk"][1] == "ТҮСІРУ"
    info = client.get("/api/info").json()["exercises"]["deadlift"]
    assert {f["key"] for f in info["fields"]} == set(DEADLIFT.default_rules())
    assert set(info["errors"]) == set(DEADLIFT_DEMO)


def test_server_rejects_bad_exercise_and_settings():
    assert client.get("/api/demo", params={"exercise": "dance"}).status_code == 400
    assert client.get("/api/demo", params={"exercise": "deadlift", "errors": "shallow"}).status_code == 400
    bad = client.get("/api/demo", params={"exercise": "deadlift", "settings": '{"max_bar_dist": 9}', "lang": "kk"})
    assert bad.status_code == 400 and "max_bar_dist" in bad.json()["detail"]


def test_websocket_deadlift():
    lm = session(2)
    with client.websocket_connect("/ws/live") as ws:
        ws.send_json({"type": "start", "lang": "kk", "exercise": "deadlift", "settings": {}})
        ready = ws.receive_json()
        assert ready["type"] == "ready" and ready["phase_names"]["ru"][0] == "ВЕРХ"
        for i in range(0, len(lm), 2):
            ws.send_json({"type": "frame", "t": i / 30, "lm": lm[i].ravel().tolist()})
            state = ws.receive_json()
        assert state["reps"] == 2
        ws.send_json({"type": "stop"})
        result = ws.receive_json()
        assert result["type"] == "result" and result["data"]["exercise"] == "deadlift"
