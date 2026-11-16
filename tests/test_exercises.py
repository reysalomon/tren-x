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


def floor_session(reps: int = 3, fps: float = 30.0) -> np.ndarray:
    """Классическая тяга с пола, как в реальных видео (датасет MyDeadlift): человек начинает в наклоне
    у штанги, встаёт, опускает штангу обратно и секунду отдыхает внизу."""
    bottom = skeleton(75, 130, bar=0.1)
    frames = [bottom] * int(fps)
    for _ in range(reps):
        for k in (1 + np.cos(np.linspace(0, 2 * np.pi, int(2.5 * fps)))) / 2:  # 1 (низ) -> 0 (стоя) -> 1 (низ)
            frames.append(skeleton(75 * k, 170 - 40 * k, bar=0.1 * k))
        frames += [bottom] * int(fps)
    return np.array(frames)


def test_video_counts_reps_from_floor():
    a = video(floor_session(3))
    assert len(a.reps) == 3
    assert all(r.good for r in a.reps), [r.errors for r in a.reps]


def test_live_counts_reps_from_floor():
    lm = floor_session(3)
    live = ExerciseLiveSession(DEADLIFT)
    for i in range(0, len(lm), 2):  # 15 кадров в секунду
        live.push(i / 30.0, lm[i])
    assert len(live.agent.reps) == 3


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
from formcoach.exercises import PUSHUP, PUSHUP_DEMO, in_plank, pushup_values  # noqa: E402


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


def test_pushup_ignores_arm_bending_while_standing():
    """Живой тест с телефона: стоя/при ходьбе локоть сгибается — это не отжимание (проверка in_plank)."""
    assert in_plank(pushup_skeleton(165), "L") and in_plank(pushup_skeleton(90), "L")
    assert not in_plank(arms_down(90), "L")
    k = (1 - np.cos(np.linspace(0, 2 * np.pi, 66))) / 2
    standing = np.array([arms_down(170)] * 30 + [arms_down(170 - 120 * x) for x in k] * 3 + [arms_down(170)] * 30)
    lm = np.concatenate([standing, pushup_session(3)])
    assert len(pushup_video(lm).reps) == 3  # три настоящих отжимания, сгибания стоя не посчитаны
    live = ExerciseLiveSession(PUSHUP)
    for i in range(len(lm)):
        live.push(i / 30.0, lm[i])
    assert len(live.agent.reps) == 3


@pytest.mark.parametrize("error", list(PUSHUP_DEMO))
def test_pushup_demo_and_server(error):
    r = client.get("/api/demo", params={"exercise": "pushup", "errors": error, "lang": "kk"})
    assert r.status_code == 200
    assert [[e["key"] for e in rep["errors"]] for rep in r.json()["reps"]] == [[], [error], []]


# ---------------- подтягивания ----------------
from formcoach.exercises import PULLUP, PULLUP_DEMO, on_bar, pullup_values  # noqa: E402


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


def arms_down(elbow: float) -> np.ndarray:
    """Человек стоит у турника и сгибает руки ВНИЗУ (разминается): та же поза, но руки отражены вниз от плеч."""
    pts = pullup_skeleton(elbow)
    sh_y = pts[P.L_SHOULDER, 1]
    for i in (P.L_ELBOW, P.R_ELBOW, P.L_WRIST, P.R_WRIST):
        pts[i, 1] = 2 * sh_y - pts[i, 1]
    return pts


def test_pullup_ignores_arm_bending_off_the_bar():
    """Правило "кисти выше плеч" (найдено на видео RepCount): сгибания рук не на перекладине — не подтягивания."""
    assert on_bar(pullup_skeleton(170), "L") and on_bar(pullup_skeleton(40), "L")
    assert not on_bar(arms_down(90), "L")
    k = (1 - np.cos(np.linspace(0, 2 * np.pi, 66))) / 2
    warmup = np.array([arms_down(170)] * 30 + [arms_down(170 - 130 * x) for x in k] * 3 + [arms_down(170)] * 30)
    lm = np.concatenate([warmup, pullup_session(3)])
    assert len(pullup_video(lm).reps) == 3  # три настоящих, разминка не посчитана
    live = ExerciseLiveSession(PULLUP)
    for i in range(len(lm)):
        live.push(i / 30.0, lm[i])
    assert len(live.agent.reps) == 3


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


# ---------------- брусья и жим лёжа ----------------
from formcoach.exercises import (BENCH, BENCH_DEMO, DIPS, DIPS_DEMO, bench_values, body_moves,  # noqa: E402
                                 dips_values, on_bars, on_bench)


def put(pts: np.ndarray, joints: dict, facing: int = 1) -> np.ndarray:
    """Одинаковые точки для левой и правой стороны (камера строго сбоку)."""
    for side in ("L", "R"):
        for name, xy in joints.items():
            pts[getattr(P, f"{side}_{name}"), :2] = xy
    return pts


def dips_skeleton(elbow: float, facing: int = 1) -> np.ndarray:
    """Человек на брусьях сбоку: кисти на брусьях (не двигаются), предплечье вертикально,
    плечо — под углом elbow к предплечью. При elbow < 90 плечо уходит ниже локтя."""
    pts = np.zeros((33, 4))
    pts[:, 3] = 0.95
    wrist = np.array([300.0, 400.0])
    el = wrist - [0, 70]
    t = np.radians(elbow)
    sh = el + 70 * np.array([-np.sin(t) * facing, np.cos(t)])  # 180 — плечо над локтем, 90 — на уровне локтя
    hip = sh + 150 * np.array([-np.sin(np.radians(15)) * facing, np.cos(np.radians(15))])  # корпус чуть вперёд
    knee = hip + [0, 90]
    ankle = knee + [-60 * facing, 40]  # голени согнуты назад
    return put(pts, {"SHOULDER": sh, "ELBOW": el, "WRIST": wrist, "HIP": hip, "KNEE": knee, "ANKLE": ankle,
                     "HEEL": ankle, "FOOT": ankle + [5 * facing, 5]})


def bench_skeleton(elbow: float, hip_rise: float = 0.0, facing: int = 1) -> np.ndarray:
    """Человек лёжа на скамье сбоку: плечо на месте, кисти над плечом, hip_rise — таз выше плеча (в длинах корпуса)."""
    pts = np.zeros((33, 4))
    pts[:, 3] = 0.95
    half = np.radians(elbow) / 2
    sh = np.array([300.0, 300.0])
    wrist = sh - [0, 2 * 70 * np.sin(half)]
    el = (sh + wrist) / 2 - [70 * np.cos(half) * facing, 0]  # локоть уходит в сторону ног
    hip = sh + [-150 * facing, -150 * hip_rise]
    knee = hip + [-70 * facing, -40]
    ankle = knee + [-10 * facing, 110]
    return put(pts, {"SHOULDER": sh, "ELBOW": el, "WRIST": wrist, "HIP": hip, "KNEE": knee, "ANKLE": ankle,
                     "HEEL": ankle, "FOOT": ankle + [5 * facing, 5]})


def press_session(make, reps: int = 3, bad: dict | None = None, fps: float = 30.0) -> np.ndarray:
    """Наверху 1 с, потом reps повторений по 2 с. bad — второе повторение: {"elbow": угол внизу,
    "lock": угол наверху перед повторением, остальное — в make (например hip_rise)}."""
    bad = bad or {}
    frames = [make(168)] * int(fps)
    for i in range(reps):
        extra = bad if i == 1 else {}
        lock = extra.get("lock", 168)
        kw = {k: v for k, v in extra.items() if k not in ("elbow", "lock")}
        frames += [make(lock)] * int(fps)
        k = (1 - np.cos(np.linspace(0, 2 * np.pi, int(2 * fps)))) / 2
        base = np.where(np.arange(len(k)) < len(k) // 2, lock, 168)
        for b, x in zip(base, k):
            frames.append(make(b - (b - extra.get("elbow", 85)) * x, **{n: v * x for n, v in kw.items()}))
    frames += [make(168)] * int(fps)
    return np.array(frames)


def run_video(spec, lm: np.ndarray):
    return analyze_exercise(spec, exercise_features(spec, P.PoseSequence(lm, 30.0)), 30.0)


def run_live(spec, lm: np.ndarray, step: int = 1):
    live = ExerciseLiveSession(spec)
    for i in range(0, len(lm), step):
        live.push(i / 30.0, lm[i])
    return live.agent.reps


def test_dips_values():
    for facing in (1, -1):
        v = dips_values(dips_skeleton(90, facing), "L", 0)
        assert v["elbow_angle"] == pytest.approx(90, abs=0.5)
        assert v["shoulder_drop"] == pytest.approx(0, abs=0.02)  # 90° — плечо на уровне локтя
        assert dips_values(dips_skeleton(60, facing), "L", 0)["shoulder_drop"] > 0.4
        assert on_bars(dips_skeleton(170, facing), "L")
    assert not on_bars(pushup_skeleton(165), "L") and not on_bars(bench_skeleton(165), "L")


@pytest.mark.parametrize("bad, code", [(None, None), ({"elbow": 125}, "shallow"), ({"elbow": 55}, "too_deep"),
                                       ({"lock": 145}, "no_lockout")])
def test_dips_video_on_skeleton(bad, code):
    a = run_video(DIPS, press_session(dips_skeleton, 3, bad))
    assert [r.errors for r in a.reps] == [[], [code] if code else [], []]


def test_dips_live_same_as_video():
    lm = press_session(dips_skeleton, 3, {"elbow": 55})
    expected = [r.errors for r in run_video(DIPS, lm).reps]
    assert [r.errors for r in run_live(DIPS, lm)] == expected
    assert [r.errors for r in run_live(DIPS, lm, step=3)] == expected  # 10 кадров/с


def test_dips_ignores_arm_bending_while_standing():
    """Стоя у брусьев руки тоже сгибаются, но двигаются кисти, а не тело — это не повторение (body_moves)."""
    k = (1 - np.cos(np.linspace(0, 2 * np.pi, 66))) / 2
    standing = np.array([arms_down(170)] * 30 + [arms_down(170 - 120 * x) for x in k] * 3 + [arms_down(170)] * 30)
    assert on_bars(arms_down(90), "L")  # поза похожа (кисти ниже плеч, корпус вертикально)...
    assert len(run_video(DIPS, standing).reps) == 0 and len(run_live(DIPS, standing)) == 0  # ...но это не брусья
    assert len(run_video(DIPS, np.concatenate([standing, press_session(dips_skeleton, 3)])).reps) == 3
    assert body_moves({"sh_y:min": 1.0, "sh_y:max": 2.0, "wr_y:min": 5.0, "wr_y:max": 5.1})
    assert not body_moves({"sh_y:min": 1.0, "sh_y:max": 1.05, "wr_y:min": 4.0, "wr_y:max": 5.0})


def test_bench_values():
    for facing in (1, -1):
        v = bench_values(bench_skeleton(90, facing=facing), "L", 0)
        assert v["elbow_angle"] == pytest.approx(90, abs=0.5)
        assert v["hip_rise"] == pytest.approx(0, abs=0.02)
        assert bench_values(bench_skeleton(165, 0.4, facing), "L", 0)["hip_rise"] == pytest.approx(0.4, abs=0.05)
        assert on_bench(bench_skeleton(165, facing=facing), "L")
    assert not on_bench(pushup_skeleton(165), "L")  # упор лёжа: кисти ниже плеч
    assert not on_bench(arms_down(90), "L")  # стоя: корпус вертикален


@pytest.mark.parametrize("bad, code", [(None, None), ({"elbow": 125}, "no_chest"), ({"hip_rise": 0.4}, "butt_up"),
                                       ({"lock": 145}, "no_lockout")])
def test_bench_video_on_skeleton(bad, code):
    a = run_video(BENCH, press_session(bench_skeleton, 3, bad))
    assert [r.errors for r in a.reps] == [[], [code] if code else [], []]


def test_bench_live_same_as_video_and_ignores_pushups():
    lm = press_session(bench_skeleton, 3, {"hip_rise": 0.4})
    expected = [r.errors for r in run_video(BENCH, lm).reps]
    assert [r.errors for r in run_live(BENCH, lm)] == expected
    assert len(run_live(BENCH, pushup_session(3))) == 0  # отжимания в режиме жима не считаются


@pytest.mark.parametrize("key, error", [("dips", e) for e in DIPS_DEMO] + [("bench", e) for e in BENCH_DEMO])
def test_dips_bench_demo_and_server(key, error):
    r = client.get("/api/demo", params={"exercise": key, "errors": error, "lang": "kk"})
    assert r.status_code == 200
    data = r.json()
    assert [[e["key"] for e in rep["errors"]] for rep in data["reps"]] == [[], [error], []]
    assert data["phase_names"]["ru"][0] == "ВВЕРХУ"


def test_info_lists_five_exercises():
    info = client.get("/api/info").json()["exercises"]
    assert set(info) == {"deadlift", "pushup", "pullup", "dips", "bench"}  # присед — отдельный модуль (agent.py)
    assert set(info["bench"]["errors"]) == set(BENCH_DEMO) and set(info["dips"]["errors"]) == set(DIPS_DEMO)
