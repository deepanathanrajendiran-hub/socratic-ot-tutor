# Socratic-OT Frontend

Next.js 14 + TypeScript + Tailwind. Hits the backend at `NEXT_PUBLIC_BACKEND_URL`.

## Pages

| Route | Purpose |
|-------|---------|
| `/tutor` | Socratic chat (default mode) |
| `/tutor/study` | Study chat (direct teaching mode) |
| `/architecture` | 5-panel pipeline visualizer (live + replay) |
| `/compare` | Same question, Socratic vs Study sequential reveal |
| `/dashboard` | Session state, weak topics, message log |

## Local dev

```bash
npm install
npm run dev   # http://localhost:3000
```

The backend must be running at the URL in `NEXT_PUBLIC_BACKEND_URL`
(default `http://localhost:8000`). Start it with:

```bash
cd ../backend
PYTHONPATH=. uvicorn api.main:app --port 8000
```

## Vercel deploy

Set the project's environment variable in the Vercel dashboard:

- `NEXT_PUBLIC_BACKEND_URL` = the Cloud Run service URL (e.g. `https://socratic-ot-XXX.run.app`)

The CORS allowlist in `backend/api/main.py` already permits any
`socratic-ot*.vercel.app` preview URL via regex.

## Stack

- Next.js 14 (app router) + TypeScript
- Tailwind CSS for styling
- Framer Motion for transitions
- `eventsource-parser` for backend SSE consumption
- `zustand` available for state management (currently only `useSession` etc. via hooks)
