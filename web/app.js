// FormCoach AI — логика интерфейса. Вся AI-часть работает на сервере (server.py -> formcoach/),
// здесь только: отправить видео, показать результат, хранить настройки.

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => [...document.querySelectorAll(sel)];
const PHASE_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]; // ВЕРХ, СПУСК, НИЗ, ПОДЪЁМ
const FALLBACK_DEFAULTS = {
  top_angle: 165, bottom_angle: 85, stay_prob: 0.9,
  max_thigh_angle: 35, max_torso_lean: 30, max_heel_lift: 0.2, min_rep_seconds: 0.8,
};

let defaults = { ...FALLBACK_DEFAULTS };
let settings = loadSettings();
let lastResult = null;
let lastDemo = null; // список ошибок демо — чтобы пересчитать при смене настроек
let lastSource = "src_video";

// ---------------- язык (қазақша / русский) ----------------
// Тексты лежат в i18n.js (STRINGS). Тексты анализа сервер присылает сразу на двух языках: {ru, kk}.
let lang = loadLang();
function loadLang() {
  try {
    const saved = localStorage.getItem("fc-lang");
    if (saved in STRINGS) return saved;
  } catch { /* приватный режим */ }
  return "kk"; // по умолчанию казахский
}
// t("ключ", {n: 3}) -> текст на текущем языке, {n} заменяется значением
const t = (key, vars = {}) =>
  (STRINGS[lang][key] ?? STRINGS.ru[key] ?? key).replace(/\{(\w+)\}/g, (_, v) => vars[v] ?? "");
// текст из ответа сервера: {ru: "...", kk: "..."} -> строка на текущем языке
const tx = (obj) => (obj && typeof obj === "object" ? obj[lang] ?? obj.ru : obj);

function applyLang() {
  document.documentElement.lang = lang;
  $$("[data-i18n]").forEach((el) => (el.innerHTML = t(el.dataset.i18n)));
  $$("[data-i18n-aria]").forEach((el) => el.setAttribute("aria-label", t(el.dataset.i18nAria)));
  $$("[data-lang]").forEach((b) => b.classList.toggle("on", b.dataset.lang === lang));
}
function setLang(next) {
  if (next === lang || !(next in STRINGS)) return;
  lang = next;
  try {
    localStorage.setItem("fc-lang", lang);
  } catch { /* не запоминаем */ }
  applyLang();
  buildSettings();
  if (lastResult) render(lastResult, lastSource, false); // результат уже есть на двух языках — пересчитывать не нужно
  if ($("#screen-about").classList.contains("active")) renderAbout();
}
$$("[data-lang]").forEach((b) => b.addEventListener("click", () => setLang(b.dataset.lang)));

// ---------------- навигация ----------------
function go(name) {
  $$(".screen").forEach((s) => s.classList.toggle("active", s.id === `screen-${name}`));
  $$(".tabbar button").forEach((b) => b.classList.toggle("active", b.dataset.go === name));
  window.scrollTo({ top: 0 });
  if (name === "about") renderAbout();
}
$$("[data-go]").forEach((b) => b.addEventListener("click", () => go(b.dataset.go)));

// ---------------- сообщения об ошибках ----------------
let toastTimer;
function toast(msg) {
  const t = $("#toast");
  t.textContent = msg;
  t.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => (t.hidden = true), 6000);
}
$("#toast").addEventListener("click", () => ($("#toast").hidden = true));

const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

// ---------------- настройки (хранятся в браузере) ----------------
function loadSettings() {
  try {
    return JSON.parse(localStorage.getItem("fc-settings")) || {};
  } catch {
    return {};
  }
}
function saveSettings() {
  try {
    localStorage.setItem("fc-settings", JSON.stringify(settings));
  } catch { /* приватный режим — просто не запоминаем */ }
}
const current = () => ({ ...defaults, ...settings });

function buildSettings() {
  const values = current();
  $$(".field").forEach((f) => {
    const { key, min, max, step, label, help } = f.dataset;
    const unit = f.dataset.unit === " s" ? ` ${t("sec")}` : f.dataset.unit;
    const id = `set-${key}`;
    f.innerHTML = `
      <div class="field-row"><label for="${id}">${t(label)}</label><output id="${id}-out"></output></div>
      ${help ? `<p class="help">${t(help)}</p>` : ""}
      <input type="range" id="${id}" min="${min}" max="${max}" step="${step}" value="${values[key]}">`;
    const input = f.querySelector("input");
    const out = f.querySelector("output");
    const show = () => (out.textContent = `${+(+input.value).toFixed(2)}${unit}`);
    show();
    input.addEventListener("input", show);
    input.addEventListener("change", () => {
      settings[key] = +input.value;
      saveSettings();
      if (lastDemo !== null) runDemo(lastDemo, false);
    });
  });
}
$("#btn-reset").addEventListener("click", () => {
  settings = {};
  saveSettings();
  buildSettings();
  if (lastDemo !== null) runDemo(lastDemo, false);
});

// ---------------- загрузка видео ----------------
function setStep(n) {
  $$("#load-steps li").forEach((li) => {
    const s = +li.dataset.step;
    li.className = s < n ? "done" : s === n ? "now" : "";
  });
}
function showLoading(on) {
  $("#loading").hidden = !on;
  document.body.style.overflow = on ? "hidden" : "";
}

function uploadVideo(file) {
  if (!file) return;
  if (!file.type.startsWith("video/") && !/\.(mp4|mov|webm|avi|mkv|3gp|m4v)$/i.test(file.name)) {
    toast(t("not_video"));
    return;
  }
  if (file.size > 100 * 1024 * 1024) {
    toast(t("too_big"));
    return;
  }
  const form = new FormData();
  // у снятых на iPhone файлов имя иногда без расширения — серверу нужно расширение
  const name = /\.[a-z0-9]+$/i.test(file.name) ? file.name : `video.${file.type.includes("quicktime") ? "mov" : "mp4"}`;
  form.append("video", file, name);
  form.append("settings", JSON.stringify(current()));
  form.append("lang", lang);

  const xhr = new XMLHttpRequest(); // XHR, а не fetch: у него есть прогресс отправки
  xhr.open("POST", "/api/analyze");
  xhr.responseType = "json";
  showLoading(true);
  $("#load-title").textContent = t("load_upload");
  $("#load-bar").classList.remove("indeterminate");
  $("#load-bar").style.width = "0";
  setStep(0);

  xhr.upload.onprogress = (e) => {
    if (e.lengthComputable) $("#load-bar").style.width = `${(e.loaded / e.total) * 100}%`;
  };
  xhr.upload.onload = () => {
    $("#load-title").textContent = t("load_analyze");
    $("#load-bar").classList.add("indeterminate");
    setStep(1);
    // поза — самый долгий шаг; через ~15 с показываем, что дошли до HMM и тренера
    setTimeout(() => !$("#loading").hidden && setStep(2), 15000);
  };
  xhr.onload = () => {
    showLoading(false);
    if (xhr.status === 200 && xhr.response) {
      lastDemo = null;
      render(xhr.response, "src_video");
    } else {
      toast(xhr.response?.detail || t("server_error", { status: xhr.status }));
    }
  };
  xhr.onerror = () => {
    showLoading(false);
    toast(t("no_connection"));
  };
  xhr.send(form);
}
["#input-camera", "#input-gallery"].forEach((sel) =>
  $(sel).addEventListener("change", (e) => {
    uploadVideo(e.target.files[0]);
    e.target.value = ""; // чтобы можно было выбрать тот же файл ещё раз
  }),
);

// ---------------- демо ----------------
$$("#demo-chips .chip").forEach((c) =>
  c.addEventListener("click", () => {
    c.classList.toggle("on");
    c.setAttribute("aria-pressed", c.classList.contains("on"));
  }),
);
$("#btn-demo").addEventListener("click", () => {
  runDemo($$("#demo-chips .chip.on").map((c) => c.dataset.err), true);
});

async function runDemo(errors, navigate) {
  const q = new URLSearchParams({ errors: errors.join(","), settings: JSON.stringify(current()), lang });
  try {
    const res = await fetch(`/api/demo?${q}`);
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail);
    lastDemo = errors;
    render(data, "src_demo", navigate);
  } catch (e) {
    toast(e.message || t("no_connection"));
  }
}

// ---------------- отрисовка результата ----------------
function render(data, source, navigate = true) {
  lastResult = data;
  lastSource = source;
  $("#result-empty").hidden = true;
  $("#result").hidden = false;

  const { reps, good, bad, seconds } = data.summary;
  const share = reps ? good / reps : 0;
  $("#score-source").textContent = t(source);
  $("#score-good").textContent = good;
  $("#score-total").textContent = reps;
  $("#st-reps").textContent = reps;
  $("#st-bad").textContent = bad;
  $("#st-time").textContent = seconds;
  $("#score-headline").textContent =
    t(reps === 0 ? "h_none" : share === 1 ? "h_perfect" : share >= 0.6 ? "h_ok" : "h_fix");
  $("#score-sub").textContent = reps === 0 ? t("sub_none") : t("sub_good", { good, reps });
  const ring = $("#ring-fg");
  ring.style.stroke = share >= 0.6 || reps === 0 ? "var(--accent)" : "var(--warn)";
  ring.style.strokeDashoffset = 327;
  requestAnimationFrame(() => requestAnimationFrame(() => (ring.style.strokeDashoffset = 327 * (1 - share))));

  renderAdvice(data.reps);
  renderReps(data.reps);
  renderChart(data);
  renderFeedback(data.feedback);
  $("#ml-note").textContent = t(data.has_ml ? "ml_note" : "ml_demo");
  if (navigate) go("result");
}

function renderAdvice(reps) {
  const counts = {};
  reps.forEach((r) => r.errors.forEach((e) => (counts[e.key] ??= { n: 0, text: tx(e.text) }).n++));
  const list = Object.values(counts).sort((a, b) => b.n - a.n);
  const ul = $("#top-advice");
  ul.classList.toggle("ok", list.length === 0);
  ul.innerHTML = list.length
    ? list.map((e) => `<li><span class="badge">${e.n}×</span><span>${esc(e.text)}</span></li>`).join("")
    : `<li>${t(reps.length ? "advice_none" : "advice_empty")}</li>`;
}

const SILHOUETTE = `<svg class="silhouette" viewBox="0 0 64 64" aria-hidden="true"><circle cx="30" cy="10" r="6"/>
  <path d="M27 18h7l6 16 12 1v6l-16-1-4 10 6 10h-7l-8-11 3-10-7-9z"/></svg>`;

function renderReps(reps) {
  const box = $("#reps");
  if (!reps.length) {
    box.innerHTML = `<div class="card muted">${t("reps_none")}</div>`;
    return;
  }
  const rules = current();
  box.innerHTML = reps.map((r) => {
    const keys = r.errors.map((e) => e.key);
    const verdict = r.good
      ? `<span class="verdict ok">${t("verdict_ok")}</span>`
      : `<span class="verdict bad">${r.errors.map((e) => esc(tx(e.short))).join(" · ")}</span>`;
    const heel = r.max_heel === null ? "—" : r.max_heel.toFixed(2);
    let ml = "";
    if (r.ml) {
      const p = r.ml.probs[r.ml.label];
      ml = `<div class="ml-line">ML: <b>${esc(tx(r.ml.label_text))}</b> ${Math.round(p * 100)}%${r.ml.no_feet ? ` · ${t("no_feet")}` : ""}
        <div class="prob ${r.ml.label === "good" ? "" : "bad"}"><i style="width:${p * 100}%"></i></div></div>`;
    }
    return `<article class="rep ${r.good ? "" : "bad"}">
      <div class="rep-img">${r.image ? `<img src="${r.image}" alt="${t("img_alt", { n: r.number })}">` : SILHOUETTE}</div>
      <div class="rep-body">
        <div class="rep-head"><span class="rep-num">#${r.number}</span>${verdict}</div>
        <div class="metrics">
          <span class="${keys.includes("shallow") ? "over" : ""}">${t("m_thigh")} <b>${Math.round(r.min_thigh)}°</b></span>
          <span>${t("m_knee")} <b>${Math.round(r.min_knee)}°</b></span>
          <span class="${keys.includes("lean") ? "over" : ""}">${t("m_lean")} <b>${Math.round(r.max_lean)}°</b></span>
          <span class="${keys.includes("heels") ? "over" : ""}">${t("m_heel")} <b>${heel}</b></span>
          <span class="${keys.includes("fast") ? "over" : ""}">${t("m_time")} <b>${r.seconds.toFixed(1)} ${t("sec")}</b></span>
          <span>${t("m_depth")} <b>${rules.max_thigh_angle}°</b></span>
        </div>
        ${ml}
      </div>
    </article>`;
  }).join("");
}

function renderChart(data) {
  const { t: times, knee, phase, bottoms } = data.series;
  const names = tx(data.phase_names);
  const W = 340, H = 200, L = 30, R = 8, T = 10, B = 22;
  const tMax = times[times.length - 1] || 1;
  const x = (v) => L + (v / tMax) * (W - L - R);
  const y = (v) => T + (1 - (v - 40) / (185 - 40)) * (H - T - B);

  // подряд идущие точки одной фазы склеиваем в полосу
  let bands = "";
  for (let i = 0; i < phase.length; ) {
    let j = i;
    while (j + 1 < phase.length && phase[j + 1] === phase[i]) j++;
    const x0 = x(times[i]), x1 = x(j + 1 < times.length ? times[j + 1] : tMax);
    bands += `<rect x="${x0}" y="${T}" width="${Math.max(x1 - x0, 0.5)}" height="${H - T - B}" fill="${PHASE_COLORS[phase[i]]}" opacity=".28"/>`;
    i = j + 1;
  }
  let grid = "";
  [60, 100, 140, 180].forEach((v) => {
    grid += `<line class="grid" x1="${L}" x2="${W - R}" y1="${y(v)}" y2="${y(v)}"/><text x="${L - 5}" y="${y(v) + 4}" text-anchor="end">${v}°</text>`;
  });
  const step = tMax > 40 ? 10 : 5;
  for (let s = 0; s <= tMax; s += step) grid += `<text x="${x(s)}" y="${H - 6}" text-anchor="middle">${s} ${t("sec")}</text>`;

  const pts = knee.map((k, i) => (k === null ? null : `${x(times[i]).toFixed(1)},${y(k).toFixed(1)}`));
  const path = pts.reduce((d, p, i) => (p === null ? d : d + (i === 0 || pts[i - 1] === null ? "M" : "L") + p), "");
  const badNums = new Set(data.reps.filter((r) => !r.good).map((r) => r.number));
  const dots = bottoms.map((b) => `<circle class="bottom-dot ${badNums.has(b.n) ? "bad" : ""}" cx="${x(b.t)}" cy="${y(b.knee)}" r="5"/>
    <text class="bottom-label" x="${x(b.t)}" y="${y(b.knee) + 18}" text-anchor="middle">${b.n}</text>`).join("");

  $("#chart").innerHTML = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${t("chart_aria")}">
      ${bands}${grid}<path class="knee" d="${path}"/>${dots}
      <line class="cursor" id="cursor" y1="${T}" y2="${H - B}" visibility="hidden"/></svg>
    <div class="chart-tip" id="chart-tip">${t("chart_hint")}</div>`;
  $("#legend").innerHTML = names.map((n, i) => `<span><i style="background:${PHASE_COLORS[i]}"></i>${n}</span>`).join("");

  // подсказка по касанию: время, угол и фаза в этой точке
  const svg = $("#chart svg");
  const onMove = (e) => {
    const box = svg.getBoundingClientRect();
    const sx = ((e.clientX - box.left) / box.width) * W;
    const tv = Math.min(Math.max(((sx - L) / (W - L - R)) * tMax, 0), tMax);
    let i = 0;
    while (i + 1 < times.length && times[i + 1] <= tv) i++;
    const c = $("#cursor");
    c.setAttribute("x1", x(times[i]));
    c.setAttribute("x2", x(times[i]));
    c.setAttribute("visibility", "visible");
    $("#chart-tip").innerHTML = `<b>${times[i].toFixed(1)} ${t("sec")}</b> · ${t("tip_angle")} <b>${knee[i] === null ? "—" : Math.round(knee[i]) + "°"}</b> · ${t("tip_phase")} <b>${names[phase[i]]}</b>`;
  };
  svg.addEventListener("pointermove", onMove);
  svg.addEventListener("pointerdown", onMove);
}

function renderFeedback(items) {
  $("#feedback").innerHTML = items.length
    ? items.map((f) => `<li class="${f.level}"><time>${f.t.toFixed(1)} ${t("sec")}</time><span class="msg">${esc(tx(f.message))}</span></li>`).join("")
    : `<li><time></time><span class="msg">${t("feedback_none")}</span></li>`;
}

// ---------------- «Как это работает» ----------------
let info = null;
async function renderAbout() {
  try {
    info ??= await (await fetch("/api/info")).json();
  } catch {
    return;
  }
  if (info.models) {
    const best = Math.max(...info.models.map((m) => m.cv_f1));
    $("#models-table").innerHTML = `<div class="table-wrap"><table class="tbl">
      <tr><th>${t("t_model")}</th><th class="n">${t("t_cv")}</th><th class="n">${t("t_test")}</th></tr>
      ${info.models.map((m) => `<tr class="${m.cv_f1 === best ? "best" : ""}"><td>${esc(m.name)}</td><td class="n">${m.cv_f1.toFixed(3)}</td><td class="n">${m.test_f1.toFixed(3)}</td></tr>`).join("")}
    </table></div>`;
  }
  let trans = lastResult?.hmm.trans;
  let phaseNames = lastResult?.phase_names;
  if (!trans) {
    try {
      const q = new URLSearchParams({ errors: "", settings: JSON.stringify(current()), lang });
      const demo = await (await fetch(`/api/demo?${q}`)).json();
      trans = demo.hmm.trans;
      phaseNames = demo.phase_names;
    } catch {
      return;
    }
  }
  const names = tx(phaseNames);
  $("#hmm-matrix").innerHTML = `<div class="table-wrap"><table class="tbl">
    <tr><th>${t("t_from_to")}</th>${names.map((n) => `<th class="n">${n}</th>`).join("")}</tr>
    ${trans.map((row, i) => `<tr><td>${names[i]}</td>${row.map((v) => `<td class="n">${v.toFixed(2)}</td>`).join("")}</tr>`).join("")}
  </table></div>`;
}

// ---------------- запуск ----------------
async function init() {
  try {
    const res = await fetch("/api/info");
    info = await res.json();
    defaults = { ...FALLBACK_DEFAULTS, ...info.defaults };
  } catch { /* сервер недоступен — покажем ошибку при первом действии */ }
  buildSettings();
}
applyLang();
init();

// PWA: офлайн-кеш интерфейса и кнопка «Установить» (Android/Chrome)
if ("serviceWorker" in navigator) navigator.serviceWorker.register("/sw.js").catch(() => {});
let installPrompt = null;
window.addEventListener("beforeinstallprompt", (e) => {
  e.preventDefault();
  installPrompt = e;
  $("#btn-install").hidden = false;
});
$("#btn-install").addEventListener("click", async () => {
  if (!installPrompt) return;
  installPrompt.prompt();
  await installPrompt.userChoice;
  installPrompt = null;
  $("#btn-install").hidden = true;
});
