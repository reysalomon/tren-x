// Service worker: запоминает файлы интерфейса, чтобы сайт открывался быстро и работал как приложение.
// Запросы к /api/ всегда идут на сервер — анализ без интернета невозможен.
const CACHE = "trenx-v16";
const SHELL = ["/", "/style.css", "/i18n.js", "/figure.js", "/app.js", "/live.js", "/manifest.webmanifest", "/icons/icon-192.png"];

self.addEventListener("install", (e) => {
  e.waitUntil(caches.open(CACHE).then((c) => c.addAll(SHELL)));
  self.skipWaiting();
});

self.addEventListener("activate", (e) => {
  e.waitUntil(caches.keys().then((keys) => Promise.all(keys.filter((k) => k !== CACHE).map((k) => caches.delete(k)))));
  self.clients.claim();
});

// Сначала сеть (чтобы всегда была свежая версия), при отсутствии сети — из кеша
self.addEventListener("fetch", (e) => {
  const url = new URL(e.request.url);
  if (e.request.method !== "GET" || url.pathname.startsWith("/api/") || url.origin !== location.origin) return;
  e.respondWith(
    fetch(e.request, { cache: "no-cache" })
      .then((res) => {
        const copy = res.clone();
        caches.open(CACHE).then((c) => c.put(e.request, copy));
        return res;
      })
      .catch(() => caches.match(e.request, { ignoreSearch: true })),
  );
});
