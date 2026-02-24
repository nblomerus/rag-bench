const CACHE = 'ragbench-shell-v2';

const OFFLINE_HTML = `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>RAG-Bench — Offline</title>
<style>
  body { background: #030712; color: #9ca3af; font-family: system-ui, sans-serif;
         display: flex; align-items: center; justify-content: center;
         height: 100vh; margin: 0; flex-direction: column; gap: 12px; text-align: center; padding: 24px; box-sizing: border-box; }
  h2   { color: #f3f4f6; margin: 0; font-size: 1.25rem; }
  p    { margin: 0; font-size: 0.875rem; max-width: 360px; }
  .dim { font-size: 0.75rem; color: #4b5563; }
  .dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%;
         background: #ef4444; animation: pulse 1.5s ease-in-out infinite; }
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.4} }
</style>
</head>
<body>
  <span class="dot"></span>
  <h2>RAG-Bench is Offline</h2>
  <p>Can't reach the server right now. Try again in a bit.</p>
  <p class="dim">Runs on a spare machine at home, so it goes down sometimes.</p>
</body>
</html>`;

// Cache index.html on install
self.addEventListener('install', event => {
    event.waitUntil(
        caches.open(CACHE).then(cache => cache.add('/'))
    );
    self.skipWaiting();
});

self.addEventListener('activate', event => {
    // Clear old caches
    event.waitUntil(
        caches.keys().then(keys =>
            Promise.all(keys.filter(k => k !== CACHE).map(k => caches.delete(k)))
        ).then(() => clients.claim())
    );
});

// For page navigation: try network (4s timeout), fall back to cache, fall back to inline HTML
self.addEventListener('fetch', event => {
    if (event.request.mode === 'navigate') {
        event.respondWith(
            fetch(event.request, { signal: AbortSignal.timeout(4000) })
                .catch(async () => {
                    const cached = await caches.match(event.request)
                        ?? await caches.match('/');
                    return cached ?? new Response(OFFLINE_HTML, {
                        headers: { 'Content-Type': 'text/html; charset=utf-8' }
                    });
                })
        );
    }
});
