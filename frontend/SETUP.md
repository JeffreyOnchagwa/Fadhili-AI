# Fadhili AI frontend — setup

I don't have direct access to your local "Fadhili AI/frontend" folder
(I run in a separate sandbox), so I built the full frontend here and
packaged it as a zip. These are complete, ready-to-drop-in files —
merge them into your existing project rather than creating a new one.

## 1. Merge the files

Unzip `fadhili-frontend.zip` and copy its contents into your existing
`Fadhili AI/frontend` folder, **replacing**:
- `index.html`
- `src/main.tsx`, `src/App.tsx`, `src/index.css`
- `tailwind.config.js`, `postcss.config.js` (new files)
- `.env.example` (new file)
- `public/favicon.svg` (new file)

and **adding** everything under `src/components`, `src/pages`,
`src/context`, `src/hooks`, `src/services`, `src/data`, `src/types`.

You can safely delete the starter's old `src/App.css` and any unused
`src/assets/react.svg` — they're no longer referenced.

## 2. Install dependencies

From `Fadhili AI/frontend`, run:

```
npm install react-router-dom lucide-react
npm install -D tailwindcss@^3 postcss autoprefixer
```

(Pinning `tailwindcss@^3` matters — v4 changes the config format, and
`tailwind.config.js`/`postcss.config.js` here are written for v3.)

## 3. Run it

```
npm run dev
```

Then open the printed local URL and check each route: Home,
Interpreter, Translate, Learn, Dictionary, About.

## 4. Build check

```
npm run build
```

I wasn't able to run `npm install`, lint, or `tsc`/`vite build` myself —
my sandbox has no network access, so it can't reach the npm registry.
I reviewed every file by hand for TypeScript correctness (strict mode,
`verbatimModuleSyntax`-safe type imports, no stray `React.*`
namespace references), but a first real build is worth running. If
`npm run build` or `npm run lint` shows errors, paste them back to me
and I'll fix them directly.

## What's real vs. honestly stubbed

- **Real**: camera access (`useCamera` hook, start/stop, permission
  and error handling, cleanup on unmount), video upload + preview,
  browser speech recognition (Web Speech API, with a feature-detection
  fallback message), browser text-to-speech for the (currently empty)
  translation field, language selection that persists via
  `localStorage`, routing, responsive layout, accessibility basics.
- **Honestly stubbed**: all recognition and sign-generation results.
  `src/services/api.ts` has real fetch calls wired up and ready — they
  just short-circuit to a "not connected" result until you set
  `VITE_API_BASE_URL` to a real FastAPI backend. No fake signs,
  confidence scores, or translations are ever generated.
- **Empty on purpose**: `src/data/dictionary.ts` — no invented KSL
  translations. The dictionary UI, filtering, and card layout are
  fully built; add verified entries to that array when you have them.

## Notable structural choices

- `src/types/speech.d.ts` adds ambient types for the Web Speech API's
  `SpeechRecognition`, which isn't in TypeScript's DOM lib by default.
- Design tokens (colors, type, radii) live in `tailwind.config.js` —
  deep teal + warm ochre + warm off-white, `Fraunces` for display type
  and `Manrope` for UI/body text, loaded from Google Fonts in
  `index.html`.
- `prefers-reduced-motion` is respected globally in `src/index.css`.
