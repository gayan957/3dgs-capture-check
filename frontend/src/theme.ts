/**
 * Light, dark, or whatever the computer says.
 *
 * The choice is kept per browser, not on the server: it is a preference about
 * this screen, and two people using the same CaptureCheck server should not
 * change each other's. `localStorage` throws in a private window with site
 * data blocked, so every read and write is guarded and the page still renders
 * when it fails -- it simply falls back to following the system.
 */

export type Theme = "system" | "light" | "dark";

export const THEMES: Theme[] = ["system", "light", "dark"];

const KEY = "capturecheck.theme";

export function readTheme(): Theme {
  try {
    const stored = localStorage.getItem(KEY);
    if (stored === "light" || stored === "dark" || stored === "system") return stored;
  } catch {
    /* blocked or unavailable: follow the system */
  }
  return "system";
}

export function storeTheme(theme: Theme): void {
  try {
    if (theme === "system") localStorage.removeItem(KEY);
    else localStorage.setItem(KEY, theme);
  } catch {
    /* the choice just will not survive a reload */
  }
}

/**
 * Put the choice on <html>, where the stylesheet reads it.
 *
 * "system" removes the attribute rather than resolving it to a value, so the
 * page keeps following the operating system if it changes while the tab is
 * open -- the `prefers-color-scheme` media query does that on its own.
 */
export function applyTheme(theme: Theme): void {
  const root = document.documentElement;
  if (theme === "system") root.removeAttribute("data-theme");
  else root.setAttribute("data-theme", theme);
  paintBrowserChrome();
}

/** Tell the browser what colour to paint its own bars around the page. */
function paintBrowserChrome(): void {
  const colour = getComputedStyle(document.documentElement)
    .getPropertyValue("--bg")
    .trim();
  if (!colour) return;
  let tag = document.querySelector<HTMLMetaElement>('meta[name="theme-color"]');
  if (!tag) {
    tag = document.createElement("meta");
    tag.name = "theme-color";
    document.head.appendChild(tag);
  }
  tag.content = colour;
}

/** What the reader is actually looking at right now. */
export function resolvedTheme(theme: Theme): "light" | "dark" {
  if (theme !== "system") return theme;
  return window.matchMedia?.("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}
