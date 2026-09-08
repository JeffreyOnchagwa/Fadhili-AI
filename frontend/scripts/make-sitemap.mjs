/**
 * Writes sitemap.xml and robots.txt into dist/ after a build.
 *
 * A sitemap needs absolute URLs, and Fadhili's deployment domain is not
 * known at authoring time, so both files are generated from
 * VITE_SITE_URL rather than committed with a hardcoded host. If that
 * variable is unset the sitemap is skipped entirely and robots.txt is
 * written without a Sitemap line — an empty or wrongly-hosted sitemap is
 * worse for crawlers than none at all.
 */

import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const dist = join(here, "..", "dist");

// Routes that render real, indexable content. Kept in sync with App.tsx.
const ROUTES = [
  "/",
  "/interpreter",
  "/translate",
  "/learn",
  "/dictionary",
  "/about",
  "/privacy",
];

const rawSite = process.env.VITE_SITE_URL?.trim();
const site = rawSite?.replace(/\/+$/, "");

const isPublic =
  site &&
  /^https?:\/\//.test(site) &&
  !/localhost|127\.0\.0\.1/.test(site);

mkdirSync(dist, { recursive: true });

if (isPublic) {
  const today = new Date().toISOString().slice(0, 10);
  const urls = ROUTES.map(
    (route) =>
      `  <url>\n    <loc>${site}${route}</loc>\n` +
      `    <lastmod>${today}</lastmod>\n  </url>`
  ).join("\n");

  writeFileSync(
    join(dist, "sitemap.xml"),
    `<?xml version="1.0" encoding="UTF-8"?>\n` +
      `<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n` +
      `${urls}\n</urlset>\n`
  );

  writeFileSync(
    join(dist, "robots.txt"),
    `# Fadhili AI\nUser-agent: *\nAllow: /\n\nSitemap: ${site}/sitemap.xml\n`
  );

  console.log(`sitemap.xml written for ${site} (${ROUTES.length} routes)`);
} else {
  writeFileSync(
    join(dist, "robots.txt"),
    `# Fadhili AI\nUser-agent: *\nAllow: /\n`
  );
  console.log(
    "VITE_SITE_URL is unset or local — wrote robots.txt, skipped sitemap.xml"
  );
}
