// TREN-X — сызба-адам (stick figure) для карточки повторения, когда кадра из видео нет (демо).
// Поза строится из измеренных углов самого повторения, поэтому у каждого повторения картинка своя:
// неглубокий присед выглядит неглубоким, провисший таз — провисшим. Часть тела с ошибкой — жёлтая.
// Координаты: x вправо (человек смотрит вправо), y вниз, как в SVG. Длины в условных единицах.

const FIG = { torso: 30, thigh: 26, shin: 26, upper: 17, fore: 16, head: 6 };
const rad = (deg) => (deg * Math.PI) / 180;
// точка на расстоянии len от p; угол 0 — вверх, +90 — вперёд (вправо), 180 — вниз
const polar = (p, len, deg) => [p[0] + len * Math.sin(rad(deg)), p[1] - len * Math.cos(rad(deg))];
const dist = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]);
const num = (v, fallback) => (typeof v === "number" && isFinite(v) ? v : fallback);

// Третья точка сустава: угол a-b-c равен angle. Решений два (в разные стороны) — выбирает prefer(c1, c2).
function bend(a, b, len, angle, prefer) {
  const base = Math.atan2(a[1] - b[1], a[0] - b[0]);
  const [c1, c2] = [1, -1].map((s) => [b[0] + len * Math.cos(base + s * rad(angle)), b[1] + len * Math.sin(base + s * rad(angle))]);
  return prefer(c1, c2) ? c1 : c2;
}

// Расстояние плечо-кисть при данном угле в локте (теорема косинусов)
const reach = (elbow, u = FIG.upper, f = FIG.fore) => Math.sqrt(u * u + f * f - 2 * u * f * Math.cos(rad(elbow)));

// Локоть по известным плечу и кисти (пересечение двух окружностей); prefer выбирает сторону сгиба
function elbowPoint(sh, wr, prefer, u = FIG.upper, f = FIG.fore) {
  const d = Math.min(Math.max(dist(sh, wr), 1e-3), u + f - 1e-3);
  const along = (u * u - f * f + d * d) / (2 * d);
  const off = Math.sqrt(Math.max(u * u - along * along, 0));
  const ex = (wr[0] - sh[0]) / d, ey = (wr[1] - sh[1]) / d;
  const m = [sh[0] + ex * along, sh[1] + ey * along];
  const c1 = [m[0] - ey * off, m[1] + ex * off], c2 = [m[0] + ey * off, m[1] - ex * off];
  return prefer(c1, c2) ? c1 : c2;
}

// голова — продолжение линии таз -> плечо
function headOn(sh, hip, gap = 10) {
  const d = dist(sh, hip) || 1;
  return [sh[0] + ((sh[0] - hip[0]) / d) * gap, sh[1] + ((sh[1] - hip[1]) / d) * gap];
}

const metric = (r) => Object.fromEntries((r.metrics ?? []).map((m, i) => [i, m.value]));

function squatPose(r, err) {
  const th = num(r.min_thigh, 10), k = num(r.min_knee, 85), lean = num(r.max_lean, 20), lift = Math.max(num(r.max_heel, 0), 0);
  const knee = [0, 0];
  const hip = [-FIG.thigh * Math.cos(rad(th)), -FIG.thigh * Math.sin(rad(th))]; // бедро под углом th к полу
  const ankle = bend(hip, knee, FIG.shin, k, (a, b) => a[1] > b[1]);
  const sh = polar(hip, FIG.torso, lean);
  const floor = ankle[1] + 4 + lift * FIG.shin * 0.8;
  const toe = [ankle[0] + 12, floor], heel = [ankle[0] - 4, floor - lift * FIG.shin * 1.6];
  const el = [sh[0] + 15, sh[1] + 6];
  return {
    floor, head: headOn(sh, hip),
    bones: [
      ["foot", [heel, toe], err.has("heels")],
      ["leg", [ankle, knee, hip], err.has("shallow")],
      ["torso", [hip, sh], err.has("lean")],
      ["arm", [sh, el, [el[0] + 15, el[1] - 3]], false],
    ],
  };
}

function deadliftPose(r, err) {
  const v = metric(r);
  const h = num(v[0], 90), k = num(v[1], 125), back = num(v[2], 0);
  const ankle = [0, 0], floor = 4;
  let knee, hip, sh, wr;
  if (err.has("lean_back")) { // эта ошибка случается наверху: стоит прямо и отклоняется назад
    knee = polar(ankle, FIG.shin, 3);
    hip = polar(knee, FIG.thigh, -3);
    sh = polar(hip, FIG.torso, -Math.max(back, 12));
    wr = [sh[0] + 5, sh[1] + 33];
  } else {
    knee = polar(ankle, FIG.shin, 12);
    hip = bend(ankle, knee, FIG.thigh, k, (a, b) => a[0] < b[0]); // таз позади колена
    sh = bend(knee, hip, FIG.torso, h, (a, b) => a[1] - a[0] / 2 < b[1] - b[0] / 2); // из двух решений — где плечи выше и впереди
    wr = err.has("bar_far") ? polar(sh, 33, 145) : [sh[0], sh[1] + 33];
    if (wr[1] > floor - 8) wr = [wr[0], floor - 8]; // блин штанги не уходит под пол
  }
  return {
    floor, head: headOn(sh, hip), bar: wr,
    bones: [
      ["foot", [[ankle[0] - 4, floor], [ankle[0] + 12, floor]], false],
      ["leg", [ankle, knee, hip], err.has("squat_it") || err.has("stiff_legs")],
      ["torso", [hip, sh], err.has("lean_back")],
      ["arm", [sh, wr], err.has("bar_far")],
    ],
  };
}

function pushupPose(r, err) {
  const v = metric(r);
  const e = num(v[0], 85), sag = num(v[1], 0), pike = num(v[2], 0);
  const floor = 0, toe = [0, 0], ankle = [2, -4], wr = [78, 0];
  const d = reach(e);
  const sh = [wr[0] - 3, -Math.sqrt(Math.max(d * d - 9, 1))];
  const el = elbowPoint(sh, wr, (a, b) => a[0] < b[0]); // локти уходят назад, к ногам
  const t = 0.62; // таз на линии лодыжка -> плечо; провис/подъём сдвигает его вниз/вверх
  const shift = Math.tan(rad(Math.min(Math.max(sag, pike), 60))) * 19 * (sag >= pike ? 1 : -1);
  const hip = [ankle[0] + (sh[0] - ankle[0]) * t, Math.min(ankle[1] + (sh[1] - ankle[1]) * t + shift, -2)];
  const knee = [(ankle[0] + hip[0]) / 2, (ankle[1] + hip[1]) / 2];
  const body = err.has("sag") || err.has("pike");
  const head = headOn(sh, hip);
  return {
    floor, head: [head[0], head[1] - 2],
    bones: [
      ["foot", [toe, ankle], false],
      ["leg", [ankle, knee, hip], body],
      ["torso", [hip, sh], body],
      ["arm", [sh, el, wr], err.has("shallow")],
    ],
  };
}

function pullupPose(r, err) {
  const v = metric(r);
  const hang = err.has("no_hang"); // ошибка про начало повторения: показываем неполный вис
  const e = hang ? num(v[2], 140) : num(v[0], 40), lean = num(v[3], 8);
  const U = FIG.upper, F = FIG.fore, wr = [0, 0];
  const sh = [-2, reach(e, U, F)];
  const el = elbowPoint(sh, wr, (a, b) => a[0] > b[0], U, F);
  const chin = hang ? null : v[1]; // рот относительно перекладины, в длинах предплечья
  const head = [4, typeof chin === "number" ? Math.min(chin * 16 - 1, sh[1] - 8) : sh[1] - 10];
  if (!hang) sh[1] = Math.min(sh[1], head[1] + 13); // шея не длиннее обычной
  const hip = polar(sh, FIG.torso, 180 - lean);
  const knee = polar(hip, FIG.thigh, 180 - lean * 0.5 - 6);
  const ankle = polar(knee, FIG.shin, 180 - lean * 0.5 + 75); // голени согнуты назад, как обычно на турнике
  return {
    head, pullbar: wr, warnHead: err.has("chin_low"),
    bones: [
      ["foot", [ankle, [ankle[0] - 3, ankle[1] + 7]], false],
      ["leg", [ankle, knee, hip], err.has("swing")],
      ["torso", [hip, sh], err.has("swing")],
      ["arm", [sh, el, wr], err.has("chin_low") || hang],
    ],
  };
}

function dipsPose(r, err) {
  const v = metric(r);
  const lock = err.has("no_lockout"); // ошибка про верх: показываем недовыпрямленные руки наверху
  const e = lock ? num(v[2], 140) : num(v[0], 85);
  const wr = [0, 0]; // кисти на брусьях
  const el = [-2, -FIG.fore]; // предплечье почти вертикально
  const sh = [el[0] + FIG.upper * Math.sin(rad(e)), el[1] + FIG.upper * Math.cos(rad(e))]; // 180 — над локтем, 90 — вровень
  const hip = polar(sh, FIG.torso, 195); // корпус чуть наклонён вперёд
  const knee = polar(hip, FIG.thigh, 185);
  const ankle = polar(knee, FIG.shin, 250); // голени согнуты назад
  return {
    head: headOn(sh, hip), pullbar: wr,
    bones: [
      ["foot", [ankle, [ankle[0] - 7, ankle[1] + 3]], false],
      ["leg", [ankle, knee, hip], false],
      ["torso", [hip, sh], false],
      ["arm", [sh, el, wr], err.has("shallow") || err.has("too_deep") || lock],
    ],
  };
}

function benchPose(r, err) {
  const v = metric(r);
  const lock = err.has("no_lockout");
  const e = lock ? num(v[2], 140) : num(v[0], 80), rise = Math.max(num(v[1], 0), 0);
  const sh = [0, 0]; // лежит на спине, голова справа
  const hip = [-FIG.torso, -rise * FIG.torso];
  const wr = [2, -reach(e)]; // гриф над плечом
  const el = elbowPoint(sh, wr, (a, b) => a[0] < b[0]); // локоть уходит к ногам
  const knee = [hip[0] - 20, hip[1] - 12];
  const ankle = [knee[0] - 3, knee[1] + 26];
  return {
    floor: ankle[1] + 2, head: [sh[0] + 9, sh[1] - 3], bar: wr,
    bones: [
      ["foot", [ankle, [ankle[0] + 8, ankle[1] + 2]], false],
      ["leg", [ankle, knee, hip], err.has("butt_up")],
      ["torso", [hip, sh], err.has("butt_up")],
      ["arm", [sh, el, wr], err.has("no_chest") || lock],
    ],
  };
}

const POSES = { squat: squatPose, deadlift: deadliftPose, pushup: pushupPose, pullup: pullupPose, dips: dipsPose, bench: benchPose };

// Готовая SVG-картинка для карточки. exercise — ключ упражнения, r — повторение из ответа сервера.
function repFigure(exercise, r) {
  const pose = (POSES[exercise] ?? squatPose)(r, new Set(r.errors.map((e) => e.key)));
  const W = 160, H = 120, PAD = 12;
  const pts = pose.bones.flatMap((b) => b[1]);
  pts.push([pose.head[0] - FIG.head, pose.head[1] - FIG.head], [pose.head[0] + FIG.head, pose.head[1] + FIG.head]);
  if (pose.bar) pts.push([pose.bar[0] - 9, pose.bar[1] - 9], [pose.bar[0] + 9, pose.bar[1] + 9]);
  if (pose.floor != null) pts.push([pts[0][0], pose.floor]);
  const xs = pts.map((p) => p[0]), ys = pts.map((p) => p[1]);
  const x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys);
  const s = Math.min((W - 2 * PAD) / Math.max(x1 - x0, 1), (H - 2 * PAD) / Math.max(y1 - y0, 1), 1.8);
  const ox = (W - (x1 - x0) * s) / 2 - x0 * s, oy = (H - (y1 - y0) * s) / 2 - y0 * s;
  const X = (p) => (p[0] * s + ox).toFixed(1), Y = (p) => (p[1] * s + oy).toFixed(1);
  const line = (b) => `<polyline class="bone${b[2] ? " warn" : ""}" points="${b[1].map((p) => `${X(p)},${Y(p)}`).join(" ")}"/>`;
  const joints = pose.bones.filter((b) => b[0] !== "foot")
    .flatMap((b) => b[1].map((p) => `<circle class="joint${b[2] ? " warn" : ""}" cx="${X(p)}" cy="${Y(p)}" r="2.3"/>`));
  let extra = "";
  if (pose.floor != null) extra += `<line class="floor" x1="10" x2="${W - 10}" y1="${Y([0, pose.floor])}" y2="${Y([0, pose.floor])}"/>`;
  if (pose.bar) extra += `<circle class="plate" cx="${X(pose.bar)}" cy="${Y(pose.bar)}" r="${(8 * s).toFixed(1)}"/>`;
  if (pose.pullbar) extra += `<line class="floor" x1="10" x2="${W - 10}" y1="${Y(pose.pullbar)}" y2="${Y(pose.pullbar)}"/>`;
  return `<svg class="fig" viewBox="0 0 ${W} ${H}" aria-hidden="true">${extra}${pose.bones.map(line).join("")}
    <circle class="skull${pose.warnHead ? " warn" : ""}" cx="${X(pose.head)}" cy="${Y(pose.head)}" r="${(FIG.head * s).toFixed(1)}"/>${joints.join("")}</svg>`;
}
