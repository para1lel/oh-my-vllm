import { createHighlighter } from "shiki";

// Match the reading surfaces while keeping small code tokens readable.
const themes = { light: "github-light", dark: "github-dark" };
const colorReplacements = {
  "github-light": {
    "#6a737d": "#576473",
    "#d73a49": "#BD293B",
    "#22863a": "#1B7534",
    "#e36209": "#AA4608",
  },
  "github-dark": { "#6a737d": "#A3B0BD" },
};

export async function createCodeHighlighter() {
  const highlighter = await createHighlighter({
    themes: Object.values(themes),
    langs: ["rust", "python", "cpp"],
  });
  return {
    highlight(text, language) {
      return highlighter.codeToTokensWithThemes(text, {
        lang: language, themes, colorReplacements,
      }).map((line) => line.map(({ content, variants }) => ({
        content,
        light: variants.light.color,
        dark: variants.dark.color,
        lightStyle: variants.light.fontStyle,
        darkStyle: variants.dark.fontStyle,
      })));
    },
    dispose() { highlighter.dispose(); },
  };
}
