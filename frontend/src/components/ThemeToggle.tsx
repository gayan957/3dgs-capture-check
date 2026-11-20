import { useEffect, useState } from "react";
import { applyTheme, readTheme, resolvedTheme, storeTheme, type Theme } from "../theme";

const NEXT: Record<Theme, Theme> = { system: "light", light: "dark", dark: "system" };
const LABEL: Record<Theme, string> = { system: "System", light: "Light", dark: "Dark" };

/** Cycles system -> light -> dark. Three states, one small button. */
export function ThemeToggle() {
  const [theme, setTheme] = useState<Theme>(readTheme);

  useEffect(() => {
    applyTheme(theme);
  }, [theme]);

  // Following the system means following it as it changes, including while
  // this tab is open.
  useEffect(() => {
    if (theme !== "system" || !window.matchMedia) return;
    const media = window.matchMedia("(prefers-color-scheme: dark)");
    const update = () => applyTheme("system");
    media.addEventListener("change", update);
    return () => media.removeEventListener("change", update);
  }, [theme]);

  const choose = (next: Theme) => {
    setTheme(next);
    storeTheme(next);
  };

  const showing = resolvedTheme(theme);

  return (
    <button
      type="button"
      className="theme-toggle"
      onClick={() => choose(NEXT[theme])}
      title={
        theme === "system"
          ? `Following your system (${showing}). Click for light.`
          : `${LABEL[theme]} mode. Click for ${LABEL[NEXT[theme]].toLowerCase()}.`
      }
      aria-label={`Colour theme: ${LABEL[theme]}. Switch to ${LABEL[NEXT[theme]]}.`}
    >
      <Icon theme={theme} />
      <span className="theme-name">{LABEL[theme]}</span>
    </button>
  );
}

function Icon({ theme }: { theme: Theme }) {
  const common = {
    viewBox: "0 0 24 24",
    width: 15,
    height: 15,
    fill: "none",
    stroke: "currentColor",
    strokeWidth: 1.7,
    strokeLinecap: "round" as const,
    strokeLinejoin: "round" as const,
    "aria-hidden": true,
  };
  if (theme === "light") {
    return (
      <svg {...common}>
        <circle cx="12" cy="12" r="4" />
        <path d="M12 2v2m0 16v2M2 12h2m16 0h2M4.9 4.9l1.4 1.4m11.4 11.4 1.4 1.4M19.1 4.9l-1.4 1.4M6.3 17.7l-1.4 1.4" />
      </svg>
    );
  }
  if (theme === "dark") {
    return (
      <svg {...common}>
        <path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8Z" />
      </svg>
    );
  }
  return (
    <svg {...common}>
      <rect x="2.5" y="4" width="19" height="13" rx="2" />
      <path d="M8 20h8" />
    </svg>
  );
}
