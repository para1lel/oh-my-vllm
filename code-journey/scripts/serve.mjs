import { createServer } from "node:http";
import { open, stat } from "node:fs/promises";
import { dirname, resolve, extname, sep } from "node:path";
import { fileURLToPath } from "node:url";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "../dist");
const port = Number(process.env.PORT || 18084);
const host = process.env.HOST || "0.0.0.0";
const types = { ".html": "text/html; charset=utf-8", ".css": "text/css; charset=utf-8",
  ".js": "text/javascript; charset=utf-8", ".woff2": "font/woff2", ".woff": "font/woff",
  ".ttf": "font/ttf", ".svg": "image/svg+xml" };
const server = createServer(async (req, res) => {
  let file;
  try {
    const path = decodeURIComponent(new URL(req.url, "http://localhost").pathname);
    file = resolve(root, "." + (path === "/" ? "/index.html" : path));
    if (!file.startsWith(root + sep)) {
      res.writeHead(403).end(); return;
    }
    if (req.method !== "GET" && req.method !== "HEAD") {
      res.writeHead(405, { Allow: "GET, HEAD" }).end(); return;
    }
    if (!(await stat(file)).isFile()) { res.writeHead(404).end(); return; }
    const handle = await open(file, "r");
    res.writeHead(200, {
      "Content-Type": types[extname(file)] || "text/plain; charset=utf-8",
      "Cache-Control": "no-cache",
      "X-Content-Type-Options": "nosniff",
    });
    if (req.method === "HEAD") { await handle.close(); res.end(); }
    else handle.createReadStream({ autoClose: true }).pipe(res);
  } catch {
    res.writeHead(404).end();
  }
});
server.on("error", (error) => { console.error(error.message); process.exit(1); });
server.listen(port, host, () => console.log("Code journey listening on http://" + host + ":" + port));
for (const signal of ["SIGTERM", "SIGINT"]) {
  process.on(signal, () => server.close(() => process.exit(0)));
}
