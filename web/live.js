// FormCoach AI — живая камера.
// Телефон сам находит 33 точки тела (MediaPipe в браузере) и по WebSocket отправляет серверу ТОЛЬКО точки,
// видео никуда не уходит. Сервер (formcoach/live.py: признаки -> HMM-фильтрация -> агент) сразу отвечает
// фазой, числом повторений и подсказками. Подсказки показываются крупно и проговариваются голосом.
// Использует из app.js: $, t, tx, lang, current, render, go, toast.

const MP_VERSION = "0.10.14";
const MP_URL = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MP_VERSION}`;
const POSE_MODEL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task";
// Голосовые фразы: web/audio/<язык>/<имя>.mp3 (записаны в ElevenLabs). Нет файла — говорит синтез речи браузера.
const CLIPS = ["start", "good", "shallow", "lean", "danger", "heels", "fast", "no_person", "finish"];
// Линии скелета на экране (номера точек MediaPipe)
const BONES = [[11, 12], [11, 23], [12, 24], [23, 24], [11, 13], [13, 15], [12, 14], [14, 16],
  [23, 25], [25, 27], [27, 29], [29, 31], [27, 31], [24, 26], [26, 28], [28, 30], [30, 32], [28, 32]];

const live = {
  landmarker: null, stream: null, ws: null, raf: 0, running: false,
  facing: "user", sound: true, t0: 0, lastVideoTime: -1, inflight: 0,
  audio: null, clips: {}, clipsLang: null, speaking: false, queue: [],
  spokenThisRep: new Set(), lastNoPerson: 0, noPersonSince: 0, hintTimer: 0, finishing: false,
};

// ---------------- экран ----------------
function liveStatus(key) {
  $("#live-status").textContent = key ? t(key) : "";
  $("#live-status").hidden = !key;
}
function showHint(text, level = "warn") {
  const h = $("#live-hint");
  h.textContent = text;
  h.className = `live-hint ${level}`;
  h.hidden = false;
  clearTimeout(live.hintTimer);
  live.hintTimer = setTimeout(() => (h.hidden = true), 3500);
}
function openLiveScreen(on) {
  $("#screen-live").hidden = !on;
  document.body.classList.toggle("live-on", on);
}

// ---------------- голос ----------------
// WebAudio, а не <audio>: на iPhone звук без нажатия пользователя играет только через уже «разбуженный» AudioContext.
async function loadClips() {
  if (live.clipsLang === lang) return;
  live.clips = {};
  live.clipsLang = lang;
  await Promise.all(CLIPS.map(async (name) => {
    try {
      const res = await fetch(`/audio/${lang}/${name}.mp3`);
      if (!res.ok) return;
      live.clips[name] = await live.audio.decodeAudioData(await res.arrayBuffer());
    } catch { /* файла нет — будет синтез речи */ }
  }));
}
function beep(freq = 880, ms = 120) {
  if (!live.sound || !live.audio) return;
  const o = live.audio.createOscillator(), g = live.audio.createGain();
  o.frequency.value = freq;
  g.gain.setValueAtTime(0.25, live.audio.currentTime);
  g.gain.exponentialRampToValueAtTime(0.001, live.audio.currentTime + ms / 1000);
  o.connect(g).connect(live.audio.destination);
  o.start();
  o.stop(live.audio.currentTime + ms / 1000);
}
function say(name, urgent = false) {
  if (!live.sound) return;
  if (urgent) live.queue = [name];
  else if (!live.queue.includes(name) && live.queue.length < 2) live.queue.push(name);
  if (!live.speaking) nextPhrase();
}
function nextPhrase() {
  const name = live.queue.shift();
  if (!name) return void (live.speaking = false);
  live.speaking = true;
  const buf = live.clips[name];
  if (buf) {
    const src = live.audio.createBufferSource();
    src.buffer = buf;
    src.connect(live.audio.destination);
    src.onended = nextPhrase;
    src.start();
    return;
  }
  // запасной вариант — синтез речи браузера (русский есть почти везде, казахский — редко)
  if ("speechSynthesis" in window) {
    const u = new SpeechSynthesisUtterance(t(`say_${name}`));
    u.lang = lang === "kk" ? "kk-KZ" : "ru-RU";
    const voice = speechSynthesis.getVoices().find((v) => v.lang.toLowerCase().startsWith(lang === "kk" ? "kk" : "ru"));
    if (voice) u.voice = voice;
    u.onend = u.onerror = nextPhrase;
    speechSynthesis.speak(u);
  } else nextPhrase();
}

// ---------------- события от тренера ----------------
function onEvent(e) {
  const text = tx(e.message);
  if (e.code === "rep") {
    beep(e.good ? 990 : 520);
    if (e.good) {
      say("good");
      showHint(text, "ok");
    } else {
      // говорим ошибку, которую ещё не сказали во время этого повторения
      const fresh = e.errors.find((x) => !live.spokenThisRep.has(x));
      if (fresh) say(fresh);
      showHint(text, "warn");
    }
    live.spokenThisRep.clear();
  } else if (e.code === "danger") {
    say("danger", true);
    live.spokenThisRep.add("lean");
    showHint(text, "danger");
  } else if (e.code !== "walking") { // lean, heels и подсказки других упражнений (bar_far, lean_back...)
    say(e.code);
    live.spokenThisRep.add(e.code);
    showHint(text, "warn");
  }
  // "walking" — без звука: это не ошибка техники
}

function onState(s) {
  $("#live-reps").textContent = s.reps;
  $("#live-good").textContent = s.good;
  $("#live-finish").disabled = false;
  const now = performance.now();
  if (s.person) {
    live.noPersonSince = 0;
    liveStatus(null);
    $("#live-phase").hidden = false;
    $("#live-phase").textContent = tx(live.phaseNames)?.[s.phase] ?? t(`ph${s.phase}`);
    $("#live-phase").dataset.phase = s.phase;
  } else {
    live.noPersonSince ||= now;
    if (now - live.noPersonSince > 1500) {
      liveStatus("live_noperson");
      $("#live-phase").hidden = true;
      if (now - live.lastNoPerson > 8000) {
        say("no_person");
        live.lastNoPerson = now;
      }
    }
  }
  s.events.forEach(onEvent);
}

// ---------------- камера и модель ----------------
async function startCamera() {
  stopCamera();
  live.stream = await navigator.mediaDevices.getUserMedia({
    video: { facingMode: live.facing, width: { ideal: 640 }, height: { ideal: 480 } },
    audio: false,
  });
  const v = $("#live-video");
  v.srcObject = live.stream;
  await v.play();
  const mirror = live.facing === "user"; // фронтальную камеру показываем зеркально, как в зеркале
  v.classList.toggle("mirror", mirror);
  $("#live-canvas").classList.toggle("mirror", mirror);
}
function stopCamera() {
  live.stream?.getTracks().forEach((tr) => tr.stop());
  live.stream = null;
}
async function loadLandmarker() {
  if (live.landmarker) return live.landmarker;
  const { FilesetResolver, PoseLandmarker } = await import(`${MP_URL}/vision_bundle.mjs`);
  const fileset = await FilesetResolver.forVisionTasks(`${MP_URL}/wasm`);
  const make = (delegate) => PoseLandmarker.createFromOptions(fileset, {
    baseOptions: { modelAssetPath: POSE_MODEL, delegate },
    runningMode: "VIDEO",
    numPoses: 1,
  });
  try {
    live.landmarker = await make("GPU");
  } catch {
    live.landmarker = await make("CPU"); // не на всех телефонах есть WebGL2
  }
  return live.landmarker;
}

// ---------------- связь с сервером ----------------
function connect() {
  return new Promise((resolve, reject) => {
    const ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws/live`);
    ws.onopen = () => ws.send(JSON.stringify({ type: "start", lang, exercise, settings: current() }));
    ws.onmessage = (m) => {
      const msg = JSON.parse(m.data);
      if (msg.type === "ready") {
        live.phaseNames = msg.phase_names ?? null; // у каждого упражнения свои названия фаз
        resolve(ws);
      }
      else if (msg.type === "state") {
        live.inflight = Math.max(0, live.inflight - 1);
        onState(msg);
      } else if (msg.type === "result") finishWithResult(msg.data);
      else if (msg.type === "error") {
        live.inflight = Math.max(0, live.inflight - 1);
        toast(msg.detail);
        if (live.finishing) closeLive();
        reject(new Error(msg.detail));
      }
    };
    ws.onerror = () => reject(new Error(t("live_lost")));
    ws.onclose = () => {
      if (live.running && !live.finishing) {
        toast(t("live_lost"));
        closeLive();
      }
    };
    live.ws = ws;
  });
}

// ---------------- главный цикл: кадр -> поза -> серверу ----------------
function loop() {
  if (!live.running) return;
  live.raf = requestAnimationFrame(loop);
  const v = $("#live-video");
  if (v.readyState < 2 || v.currentTime === live.lastVideoTime) return; // новый кадр ещё не пришёл
  live.lastVideoTime = v.currentTime;
  const now = performance.now();
  const res = live.landmarker.detectForVideo(v, now);
  const pts = res.landmarks?.[0];
  drawSkeleton(pts, v.videoWidth, v.videoHeight);
  // если сервер не успевает отвечать (медленный интернет) — пропускаем кадр, а не копим очередь
  if (live.ws?.readyState !== WebSocket.OPEN || live.inflight > 2 || live.finishing) return;
  // x, y — в пикселях кадра (как в Python для видео), z и visibility — как есть
  const lm = pts ? pts.flatMap((p) => [p.x * v.videoWidth, p.y * v.videoHeight, p.z, p.visibility ?? 0]) : null;
  live.ws.send(JSON.stringify({ type: "frame", t: (now - live.t0) / 1000, lm }));
  live.inflight++;
}

function drawSkeleton(pts, w, h) {
  const c = $("#live-canvas");
  if (c.width !== w || c.height !== h) {
    c.width = w;
    c.height = h;
  }
  const ctx = c.getContext("2d");
  ctx.clearRect(0, 0, w, h);
  if (!pts) return;
  const ok = (p) => (p.visibility ?? 1) > 0.5;
  ctx.lineWidth = Math.max(3, w / 120);
  ctx.lineCap = "round";
  ctx.strokeStyle = "#c6ff3d";
  for (const [a, b] of BONES) {
    if (!ok(pts[a]) || !ok(pts[b])) continue;
    ctx.beginPath();
    ctx.moveTo(pts[a].x * w, pts[a].y * h);
    ctx.lineTo(pts[b].x * w, pts[b].y * h);
    ctx.stroke();
  }
  ctx.fillStyle = "#ffffff";
  for (const i of [11, 12, 23, 24, 25, 26, 27, 28]) {
    if (!ok(pts[i])) continue;
    ctx.beginPath();
    ctx.arc(pts[i].x * w, pts[i].y * h, ctx.lineWidth * 1.2, 0, 2 * Math.PI);
    ctx.fill();
  }
}

// ---------------- старт / стоп ----------------
async function startLive() {
  if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia) {
    toast(t("live_https"));
    return;
  }
  // AudioContext создаём по нажатию кнопки — иначе браузер не разрешит звук
  live.audio ??= new (window.AudioContext || window.webkitAudioContext)();
  live.audio.resume();
  Object.assign(live, { finishing: false, inflight: 0, lastVideoTime: -1, queue: [], speaking: false, noPersonSince: 0 });
  live.spokenThisRep.clear();
  $("#live-reps").textContent = "0";
  $("#live-good").textContent = "0";
  $("#live-finish").disabled = true;
  $("#live-phase").hidden = true;
  $("#live-hint").hidden = true;
  openLiveScreen(true);
  try {
    liveStatus("live_cam");
    await startCamera();
  } catch (e) {
    closeLive();
    toast(t(e.name === "NotAllowedError" ? "live_denied" : "live_nocam"));
    return;
  }
  try {
    liveStatus("live_model");
    await Promise.all([loadLandmarker(), loadClips()]);
  } catch {
    closeLive();
    toast(t("live_model_fail"));
    return;
  }
  try {
    liveStatus("live_connect");
    await connect();
  } catch (e) {
    closeLive();
    toast(e.message || t("live_lost"));
    return;
  }
  liveStatus("live_ready");
  say("start");
  live.t0 = performance.now();
  live.running = true;
  loop();
}

function finishLive() {
  if (!live.ws || live.ws.readyState !== WebSocket.OPEN) return closeLive();
  live.finishing = true;
  $("#live-finish").disabled = true;
  liveStatus("live_finishing");
  live.ws.send(JSON.stringify({ type: "stop" }));
}

function finishWithResult(data) {
  say("finish");
  closeLive();
  lastDemo = null;
  render(data, "src_live");
}

function closeLive() {
  live.running = false;
  cancelAnimationFrame(live.raf);
  stopCamera();
  if (live.ws) {
    live.ws.onclose = null;
    live.ws.close();
    live.ws = null;
  }
  openLiveScreen(false);
}

$("#btn-live").addEventListener("click", startLive);
$("#live-finish").addEventListener("click", finishLive);
$("#live-close").addEventListener("click", closeLive);
$("#live-sound").addEventListener("click", () => {
  live.sound = !live.sound;
  $("#live-sound").setAttribute("aria-pressed", live.sound);
  $("#live-sound").classList.toggle("off", !live.sound);
  if (!live.sound && "speechSynthesis" in window) speechSynthesis.cancel();
});
$("#live-flip").addEventListener("click", async () => {
  live.facing = live.facing === "user" ? "environment" : "user";
  try {
    await startCamera();
    live.lastVideoTime = -1;
  } catch {
    toast(t("live_nocam"));
  }
});
// ушли со страницы (свернули браузер) — выключаем камеру
document.addEventListener("visibilitychange", () => {
  if (document.hidden && live.running) closeLive();
});
