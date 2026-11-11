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


# ---------------- отжимания ----------------
from formcoach.exercises import PUSHUP, PUSHUP_DEMO, pushup_values  # noqa: E402


def pushup_skeleton(elbow: float, hip_shift: float = 0.0, facing: int = 1) -> np.ndarray:
    """Человек в упоре лёжа сбоку. elbow — угол локтя; hip_shift — таз ниже (+) или выше (-) прямой линии, пиксели."""
    pts = np.zeros((33, 4))
    pts[:, 3] = 0.95
    half = np.radians(elbow) / 2
    wrist = np.array([300 + 100 * facing, 520.0])
    sh = wrist - [0, 2 * 70 * np.sin(half)]  # плечо над кистью, высота зависит от угла локтя
    el = (sh + wrist) / 2 - [70 * np.cos(half) * facing, 0]  # локоть уходит назад, к ногам
    ankle = np.array([300 - 200 * facing, 510.0])
    hip = (sh + ankle) / 2 + [0, hip_shift]
    for side in ("L", "R"):
        g = lambda name: getattr(P, f"{side}_{name}")  # noqa: E731
        for name, xy in [("SHOULDER", sh), ("ELBOW", el), ("WRIST", wrist), ("HIP", hip), ("ANKLE", ankle),
                         ("KNEE", (hip + ankle) / 2), ("HEEL", ankle), ("FOOT", ankle + [10 * facing, 5])]:
            pts[g(name), :2] = xy
    return pts


def pushup_session(reps: int = 3, bad: dict | None = None, fps: float = 30.0) -> np.ndarray:
    """Стоит в упоре 1 с, потом reps отжиманий по 2 с; bad — второе повторение: {"elbow": .., "hip": ..}."""
    frames = [pushup_skeleton(165)] * int(fps)
    for i in range(reps):
        extra = bad if (bad and i == 1) else {}
        for k in (1 - np.cos(np.linspace(0, 2 * np.pi, int(2 * fps)))) / 2:
            frames.append(pushup_skeleton(165 - (165 - extra.get("elbow", 85)) * k, extra.get("hip", 0) * k))
        frames += [pushup_skeleton(165)] * int(fps)
    return np.array(frames)


def pushup_video(lm: np.ndarray):
    return analyze_exercise(PUSHUP, exercise_features(PUSHUP, P.PoseSequence(lm, 30.0)), 30.0)


def test_pushup_values():
    for facing in (1, -1):
        v = pushup_values(pushup_skeleton(90, facing=facing), "L", 0)
        assert v["elbow_angle"] == pytest.approx(90, abs=0.5)
        assert v["sag"] == pytest.approx(0, abs=0.5) and v["pike"] == pytest.approx(0, abs=0.5)
        assert pushup_values(pushup_skeleton(160, 30, facing), "L", 0)["sag"] > 10
        assert pushup_values(pushup_skeleton(160, -30, facing), "L", 0)["pike"] > 10


@pytest.mark.parametrize("bad, code", [(None, None), ({"elbow": 125}, "shallow"), ({"hip": 40}, "sag"),
                                       ({"hip": -45}, "pike")])
def test_pushup_video_on_skeleton(bad, code):
    a = pushup_video(pushup_session(3, bad))
    assert len(a.reps) == 3
    expected = [[], [code] if code else [], []]
    assert [r.errors for r in a.reps] == expected


@pytest.mark.parametrize("fps", [30, 10])
def test_pushup_live_same_as_video(fps):
    lm = pushup_session(3, {"hip": 40})
    expected = [r.errors for r in pushup_video(lm).reps]
    live = ExerciseLiveSession(PUSHUP)
    for i in range(0, len(lm), 30 // fps):
        live.push(i / 30.0, lm[i])
    assert [r.errors for r in live.agent.reps] == expected


@pytest.mark.parametrize("error", list(PUSHUP_DEMO))
def test_pushup_demo_and_server(error):
    r = client.get("/api/demo", params={"exercise": "pushup", "errors": error, "lang": "kk"})
    assert r.status_code == 200
    assert [[e["key"] for e in rep["errors"]] for rep in r.json()["reps"]] == [[], [error], []]


# ---------------- подтягивания ----------------
from formcoach.exercises import PULLUP, PULLUP_DEMO, pullup_values  # noqa: E402


def pullup_skeleton(elbow: float, lean: float = 5.0, facing: int = 1) -> np.ndarray:
    """Человек на турнике сбоку: кисти на перекладине (y=100), elbow — угол локтя, lean — наклон корпуса."""
    pts = np.zeros((33, 4))
    pts[:, 3] = 0.95
    half = np.radians(elbow) / 2
    wrist = np.array([300.0, 100.0])
    sh = wrist + [0, 2 * 70 * np.sin(half)]  # плечо под кистью: чем сильнее согнут локоть, тем ближе к перекладине
    el = (sh + wrist) / 2 + [70 * np.cos(half) * facing, 0]  # локоть уходит вперёд
    mouth = sh + [5 * facing, -56]  # рот выше плеча
    r = np.radians(lean)
    hip = sh + 150 * np.array([-np.sin(r) * facing, np.cos(r)])
    knee, ankle = hip + [0, 100], hip + [0, 200]
    pts[P.MOUTH_L, :2] = pts[P.MOUTH_R, :2] = pts[P.NOSE, :2] = mouth
    for side in ("L", "R"):
        g = lambda name: getattr(P, f"{side}_{name}")  # noqa: E731
        for name, xy in [("SHOULDER", sh), ("ELBOW", el), ("WRIST", wrist), ("HIP", hip), ("KNEE", knee),
                         ("ANKLE", ankle), ("HEEL", ankle), ("FOOT", ankle + [10 * facing, 0])]:
            pts[g(name), :2] = xy
    return pts


def pullup_session(reps: int = 3, bad: dict | None = None, fps: float = 30.0) -> np.ndarray:
    """Вис 1 с, потом reps подтягиваний по 2.2 с. bad — второе повторение:
    {"elbow": угол наверху}, {"lean": раскачка}, {"hang": угол виса перед ним}."""
    frames = []
    for i in range(reps):
        extra = bad if (bad and i == 1) else {}
        hang = extra.get("hang", 170)
        frames += [pullup_skeleton(hang)] * int(fps)
        for j, k in enumerate((1 - np.cos(np.linspace(0, 2 * np.pi, int(2.2 * fps)))) / 2):
            start = hang if j < int(1.1 * fps) else 170
            frames.append(pullup_skeleton(start - (start - extra.get("elbow", 40)) * k, 5 + (extra.get("lean", 5) - 5) * k))
    frames += [pullup_skeleton(170)] * int(fps)
    return np.array(frames)


def pullup_video(lm: np.ndarray):
    return analyze_exercise(PULLUP, exercise_features(PULLUP, P.PoseSequence(lm, 30.0)), 30.0)


def test_pullup_values():
    v = pullup_values(pullup_skeleton(40), "L", 0)
    assert v["elbow_angle"] == pytest.approx(40, abs=0.5)
    assert v["chin_gap"] < 0  # рот над перекладиной
    assert pullup_values(pullup_skeleton(170), "L", 0)["chin_gap"] > 1  # в висе — далеко внизу
    assert pullup_values(pullup_skeleton(90, lean=30), "L", 0)["lean"] == pytest.approx(30, abs=0.5)


@pytest.mark.parametrize("bad, code", [(None, None), ({"elbow": 75}, "chin_low"), ({"hang": 135}, "no_hang"),
                                       ({"lean": 40}, "swing")])
def test_pullup_video_on_skeleton(bad, code):
    a = pullup_video(pullup_session(3, bad))
    assert [r.errors for r in a.reps] == [[], [code] if code else [], []]


def test_pullup_live_same_as_video():
    lm = pullup_session(3, {"elbow": 75})
    expected = [r.errors for r in pullup_video(lm).reps]
    live = ExerciseLiveSession(PULLUP)
    for i in range(0, len(lm), 2):
        live.push(i / 30.0, lm[i])
    assert [r.errors for r in live.agent.reps] == expected


@pytest.mark.parametrize("error", list(PULLUP_DEMO))
def test_pullup_demo_and_server(error):
    r = client.get("/api/demo", params={"exercise": "pullup", "errors": error, "lang": "kk"})
    assert r.status_code == 200
    data = r.json()
    assert [[e["key"] for e in rep["errors"]] for rep in data["reps"]] == [[], [error], []]
    assert data["phase_names"]["ru"][0] == "ВИС"


# ---------------- стойка на руках ----------------
from formcoach.exercises import HANDSTAND, HANDSTAND_DEMO, handstand_values  # noqa: E402


def rot(v, deg):
    a = np.radians(deg)
    return np.array([v[0] * np.cos(a) - v[1] * np.sin(a), v[0] * np.sin(a) + v[1] * np.cos(a)])


def handstand_skeleton(turn: float, elbow: float = 176, bend: float = 4, shoulder: float = 172,
                       knee: float = 176) -> np.ndarray:
    """Человек сбоку с поднятыми руками, повёрнутый на turn градусов (0 — стоит на ногах, 180 — на руках).
    Сначала строим тело в обычных координатах (y вверх, плечо в нуле), потом поворачиваем и переводим в пиксели."""
    down = np.array([0.0, -1.0])
    hip = 120 * down
    knee_pt = hip + 100 * rot(down, bend)
    ankle = knee_pt + 100 * rot(down, bend + (180 - knee))
    el = 70 * rot(down, shoulder)  # угол локоть-плечо-таз = shoulder
    wr = el + 70 * rot(el / 70, -(180 - elbow))
    pts = np.zeros((33, 4))
    pts[:, 3] = 0.95
    to_px = lambda v: np.array([400, 400]) + rot(v, turn) * [1, -1]  # noqa: E731  (y в картинке вниз)
    for side in ("L", "R"):
        g = lambda name: getattr(P, f"{side}_{name}")  # noqa: E731
        for name, xy in [("SHOULDER", np.zeros(2)), ("ELBOW", el), ("WRIST", wr), ("HIP", hip), ("KNEE", knee_pt),
                         ("ANKLE", ankle), ("HEEL", ankle), ("FOOT", ankle + [10, 0])]:
            pts[g(name), :2] = to_px(xy)
    return pts


def handstand_session(attempts: int = 3, bad: dict | None = None, hold: float = 4.0, fps: float = 30.0) -> np.ndarray:
    """Стоит 1 с, потом attempts попыток: мах 0.8 с (нога согнута) -> стойка hold с -> спуск 0.8 с -> стоит 1 с.
    bad — вторая попытка: {"elbow": ...}, {"bend": ...}, {"shoulder": ...}, {"knee": ...}, {"hold": секунды}."""
    frames = [handstand_skeleton(0)] * int(fps)
    move = (1 - np.cos(np.linspace(0, np.pi, int(0.8 * fps)))) / 2
    for i in range(attempts):
        extra = dict(bad) if (bad and i == 1) else {}
        secs = extra.pop("hold", hold)
        frames += [handstand_skeleton(178 * k, knee=176 - 70 * np.sin(np.pi * k)) for k in move]
        frames += [handstand_skeleton(178, **extra)] * int(secs * fps)
        frames += [handstand_skeleton(178 * (1 - k), knee=176 - 70 * np.sin(np.pi * k)) for k in move]
        frames += [handstand_skeleton(0)] * int(fps)
    return np.array(frames)


def handstand_video(lm: np.ndarray):
    return analyze_exercise(HANDSTAND, exercise_features(HANDSTAND, P.PoseSequence(lm, 30.0)), 30.0)


def test_handstand_values():
    v = handstand_values(handstand_skeleton(180, bend=0), "L", 0)
    assert v["stand_angle"] == pytest.approx(0, abs=2)  # вверх ногами (колено чуть согнуто — линия до лодыжки слегка наклонена)
    assert v["elbow_angle"] == pytest.approx(176, abs=0.5)
    assert v["shoulder_angle"] == pytest.approx(172, abs=0.5)
    assert v["knee_angle"] == pytest.approx(176, abs=0.5)
    assert handstand_values(handstand_skeleton(0, bend=0), "L", 0)["stand_angle"] == pytest.approx(180, abs=2)
    assert handstand_values(handstand_skeleton(180, bend=30), "L", 0)["body_bend"] > 20


@pytest.mark.parametrize("bad, code", [(None, None), ({"elbow": 135}, "bent_arms"), ({"bend": 35}, "banana"),
                                       ({"shoulder": 130}, "closed_shoulders"), ({"knee": 130}, "bent_knees"),
                                       ({"hold": 1.5}, "short_hold")])
def test_handstand_video_on_skeleton(bad, code):
    """Согнутая при махе нога НЕ считается ошибкой: оцениваем только саму стойку."""
    a = handstand_video(handstand_session(3, bad))
    assert [r.errors for r in a.reps] == [[], [code] if code else [], []]
    assert a.reps[0].values["hold_seconds:max"] == pytest.approx(4.0, abs=0.6)


def test_handstand_live_same_as_video():
    lm = handstand_session(3, {"bend": 35})
    expected = [r.errors for r in handstand_video(lm).reps]
    live = ExerciseLiveSession(HANDSTAND)
    for i in range(0, len(lm), 2):
        live.push(i / 30.0, lm[i])
    assert [r.errors for r in live.agent.reps] == expected
    assert [f.code for f in live.feedback if f.code != "rep"] == ["banana"]  # подсказка прозвучала во время стойки


@pytest.mark.parametrize("error", list(HANDSTAND_DEMO))
def test_handstand_demo_and_server(error):
    r = client.get("/api/demo", params={"exercise": "handstand", "errors": error, "lang": "kk"})
    assert r.status_code == 200
    data = r.json()
    assert [[e["key"] for e in rep["errors"]] for rep in data["reps"]] == [[], [error], []]
    assert data["phase_names"]["ru"][2] == "СТОЙКА"
    assert data["feedback"][-1]["message"]["ru"].startswith("Попытка 3")
    assert data["reps"][0]["metrics"][0]["unit"] == "s"
