@AGENTS.md

## PageSpeaks Frontend Context

**Stack:** Next.js 16 + React 19 + Tailwind v4 + TypeScript. Everything is `"use client"` — no server components, no `use server`.

**Tailwind v4:** No `tailwind.config.js`. Config is in `app/globals.css` via `@theme` / CSS variables. Do not create a tailwind config file.

**Commands:**
```bash
npm run dev      # → http://localhost:3000
npm run build    # production build (run before deploying)
npm run lint     # ESLint
```

**Required env var** (create `web/.env.local`):
```
NEXT_PUBLIC_TTS_URL=https://hydrazinev--osho-tts-oshotts-web.modal.run
```

**Key files:**
- `app/page.tsx` — entire UI (single file, ~600 lines)
- `app/globals.css` — design tokens (colors, fonts)
- `app/layout.tsx` — Playfair Display + Inter fonts

**Do not add:**
- Server components or `use server` actions
- `tailwind.config.js` or `tailwind.config.ts`
- File upload UI (removed intentionally)
- Custom voice / XTTS endpoints (removed intentionally)
