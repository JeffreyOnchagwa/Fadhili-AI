/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        ink: {
          DEFAULT: "#14181A",
          soft: "#3A4245",
        },
        paper: {
          DEFAULT: "#FBFAF6",
          dim: "#F2EFE7",
        },
        teal: {
          50: "#EEF5F2",
          100: "#D2E5DD",
          300: "#7FB3A0",
          500: "#2F6F5E",
          600: "#255A4C",
          700: "#1B4438",
          900: "#0F2921",
        },
        ochre: {
          100: "#F3E7C9",
          300: "#DFC286",
          500: "#C89B3C",
          700: "#9C7628",
        },
        signal: {
          red: "#B3423A",
        },
      },
      fontFamily: {
        display: ["'Fraunces'", "serif"],
        sans: ["'Manrope'", "system-ui", "sans-serif"],
      },
      borderRadius: {
        xl2: "1.25rem",
      },
      boxShadow: {
        soft: "0 1px 2px rgba(20, 24, 26, 0.06), 0 8px 24px -12px rgba(20, 24, 26, 0.15)",
      },
      keyframes: {
        "flow-dot": {
          "0%": { transform: "translateX(0)", opacity: "0" },
          "10%": { opacity: "1" },
          "90%": { opacity: "1" },
          "100%": { transform: "translateX(100%)", opacity: "0" },
        },
      },
      animation: {
        "flow-dot": "flow-dot 2.4s ease-in-out infinite",
      },
    },
  },
  plugins: [],
};
