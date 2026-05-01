import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        background: "var(--background)",
        foreground: "var(--foreground)",
        ivory: {
          50:  "#faf9f6",
          100: "#f4f1ea",
          200: "#e6e2d8",
          300: "#d6d0c0",
          400: "#a8a194",
          500: "#6b675f",
          600: "#4a473f",
          700: "#2e2c27",
          800: "#1f1d1b",
          900: "#15140f",
        },
        coral: {
          50:  "#fbf1ec",
          100: "#f4e1d6",
          200: "#ecc8b3",
          300: "#e0a587",
          400: "#d28b66",
          500: "#cc785c",
          600: "#b15c41",
          700: "#8c4733",
          800: "#6b3724",
          900: "#4a261a",
        },
        ink: "#1f1d1b",
      },
      fontFamily: {
        sans:  ["var(--font-geist-sans)", "system-ui", "-apple-system",
                "Segoe UI", "Roboto", "sans-serif"],
        serif: ["var(--font-source-serif)", "ui-serif", "Georgia", "Cambria",
                "Times New Roman", "serif"],
        mono:  ["var(--font-geist-mono)", "ui-monospace", "Menlo", "monospace"],
      },
      maxWidth: {
        "reading": "48rem",  // ~768px — comfortable chat reading column
      },
      keyframes: {
        "fade-up": {
          "0%":   { opacity: "0", transform: "translateY(6px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
        "blink": {
          "0%, 50%":      { opacity: "1" },
          "50.01%, 100%": { opacity: "0" },
        },
      },
      animation: {
        "fade-up": "fade-up 240ms ease-out",
        "blink":   "blink 1.05s steps(2) infinite",
      },
    },
  },
  plugins: [],
};
export default config;
