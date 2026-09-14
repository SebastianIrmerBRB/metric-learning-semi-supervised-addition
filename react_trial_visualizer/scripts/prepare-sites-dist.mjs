import {
  cpSync,
  existsSync,
  mkdirSync,
  rmSync,
  writeFileSync,
} from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const projectRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const exportedSite = path.join(projectRoot, "out");
const buildDirectory = path.join(projectRoot, "dist");
const clientDirectory = path.join(buildDirectory, "client");
const serverDirectory = path.join(buildDirectory, "server");

if (!existsSync(path.join(exportedSite, "index.html"))) {
  throw new Error("The static Next.js export is missing out/index.html.");
}

rmSync(buildDirectory, { recursive: true, force: true });
mkdirSync(serverDirectory, { recursive: true });
cpSync(exportedSite, clientDirectory, { recursive: true });

const workerSource = `const SOCIAL_IMAGE_PLACEHOLDER = "https://trial-atlas.local/og.png";

function requestForPath(request, pathname) {
  const url = new URL(request.url);
  url.pathname = pathname;
  url.search = "";
  return new Request(url, {
    method: request.method === "HEAD" ? "HEAD" : "GET",
    headers: request.headers,
  });
}

async function withRequestOrigin(response, request) {
  const contentType = response.headers.get("content-type") ?? "";
  if (!contentType.toLowerCase().includes("text/html")) return response;

  const socialImage = new URL("/og.png", request.url).href;
  const html = (await response.text()).split(SOCIAL_IMAGE_PLACEHOLDER).join(socialImage);
  const headers = new Headers(response.headers);
  headers.set("content-type", "text/html; charset=utf-8");
  headers.delete("content-length");
  return new Response(html, { status: response.status, headers });
}

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    let response = await env.ASSETS.fetch(request);

    if (
      response.status === 404 &&
      (request.method === "GET" || request.method === "HEAD") &&
      !url.pathname.split("/").at(-1)?.includes(".")
    ) {
      response = await env.ASSETS.fetch(requestForPath(request, "/index.html"));
    }

    return withRequestOrigin(response, request);
  },
};
`;

writeFileSync(path.join(serverDirectory, "index.js"), workerSource);
