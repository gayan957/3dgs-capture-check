import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

const API = process.env.CAPTURECHECK_API ?? "http://127.0.0.1:8000";

// The API runs as a separate Python process. In development Vite proxies to it
// so the browser sees one origin; in production `capturecheck serve` serves
// this build itself and the proxy never applies.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: API,
        changeOrigin: true,
        configure: (proxy) => {
          // Without this, a stopped API prints a bare ECONNREFUSED stack trace
          // on every poll, which reads like the front end has crashed. It has
          // not: it is talking to something that is not there.
          let warned = false;
          proxy.on("error", (err, _req, res) => {
            const refused = (err as NodeJS.ErrnoException).code === "ECONNREFUSED";
            if (refused && !warned) {
              warned = true;
              console.log(
                `\n  \x1b[33mThe CaptureCheck API is not running at ${API}.\x1b[0m\n` +
                  `  Start it in another terminal:  \x1b[1mcapturecheck serve\x1b[0m\n` +
                  `  (or \x1b[1mmake serve\x1b[0m). Point elsewhere with CAPTURECHECK_API=...\n`,
              );
            } else if (!refused) {
              console.log(`  [api proxy] ${err.message}`);
            }
            // Answer the browser in the shape the page already knows how to
            // show, instead of letting the request hang until it times out.
            if (res && "writeHead" in res && !res.headersSent) {
              res.writeHead(503, { "Content-Type": "application/json" });
              res.end(
                JSON.stringify({
                  detail: refused
                    ? `The CaptureCheck server is not running at ${API}. Start it with \`capturecheck serve\`.`
                    : err.message,
                }),
              );
            }
          });
          // One line back when it returns, so the recovery is visible too.
          proxy.on("proxyRes", () => {
            if (warned) {
              warned = false;
              console.log(`\n  \x1b[32mThe CaptureCheck API is back.\x1b[0m\n`);
            }
          });
        },
      },
    },
  },
  build: { outDir: "dist", sourcemap: false },
});
