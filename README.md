# Travel Concierge — AI-Powered Trip Planning System

> **Full-stack agentic travel planner** — LangChain ReAct · GPT-4 · LambdaMART · DP Scheduler · Next.js · FastAPI · AWS EC2

---

## Overview

Travel Concierge is an end-to-end AI travel planning system that turns a destination and a set of interests into a personalised, constraint-aware itinerary in seconds. It combines a multi-agent LLM backend with two custom ML algorithms — a Learning-to-Rank model for place personalisation and a Dynamic Programming scheduler for optimal itinerary construction.

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│  Next.js 14 Frontend  (TypeScript + Tailwind)       │
│  Landing page · Chat UI · Weather widget · Sessions │
└───────────────────────┬─────────────────────────────┘
                        │ REST API
                        ▼
┌─────────────────────────────────────────────────────┐
│  FastAPI Backend                                    │
│  ┌─────────────────┐   ┌──────────────────────────┐ │
│  │ Intent          │   │ TravelAgent              │ │
│  │ Classifier      │──▶│ LangChain ReAct + GPT-4 │  │
│  │ (GPT JSON)      │   │ Self-Critique Loop       │ │
│  └─────────────────┘   └──────────┬───────────────┘ │
│                                   │                 │
│  ┌────────────────────────────────▼──────────────┐  │
│  │  ML Layer                                     │  │
│  │  LambdaMART (XGBoost rank:ndcg)               │  │
│  │  Weighted Interval Scheduling DP (O(n log n)) │  │
│  └───────────────────────────────────────────────┘  │
│                                                     │
│  ┌────────────────────────────────────────────────┐ │
│  │  Services                                      │ │
│  │  Geocoding · Places · Routing · Weather        │ │
│  └────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────┘
```

---

## Key Features

### 1. LangChain ReAct Agent with Self-Critique Loop
The core agent uses LangChain's ReAct framework with GPT-4. After generating an itinerary, a critic LLM validates it for logical errors (impossible timings, wrong city attractions) and triggers a correction round automatically. Maximum 2 correction rounds to prevent infinite loops.

### 2. LambdaMART Learning-to-Rank
Replaces static rating-based sorting with a personalised ranking model trained on implicit user feedback (accept/skip events). Each place is represented as a 9-dimensional feature vector:
- Rating (normalised)
- Review count (log-normalised)
- Price level
- Category match score (token overlap with user interests)
- Distance from trip centre (Haversine)
- Time-of-day suitability (morning / afternoon / evening)
- Historical accept rate for this category

The model uses XGBoost's `rank:ndcg` objective (LambdaMART algorithm). Falls back to a weighted heuristic during cold start (< 10 feedback events).

### 3. Constraint-Aware DP Itinerary Scheduler
Implements Weighted Interval Scheduling DP (O(n log n)) to select the optimal subset of places that maximises total LTR relevance scores while respecting:
- Opening and closing hours per place
- Travel time buffers between stops (Haversine-estimated)
- Hard time window (08:00 – 22:00)
- Mandatory lunch (12:00–14:00) and dinner (18:30–20:30) breaks
- Maximum 8 stops per day

### 4. Structured LLM Intent Classifier
Replaces keyword matching with a GPT-3.5-turbo call that returns structured JSON across 6 intent categories with graceful heuristic fallback on network failure.

### 5. TSP Route Optimisation
Greedy nearest-neighbour TSP reorders multi-stop waypoints by Haversine distance before calling Google Directions API, minimising total travel distance.

---

## Project Structure

```
travel-concierge/
├── frontend/                     # Next.js 14 app
│   ├── app/
│   │   ├── page.tsx              # Landing page
│   │   ├── chat/page.tsx         # Chat interface
│   │   └── globals.css           # Design tokens
│   └── lib/api.ts                # API client
│
└── backend/
    ├── api.py                    # FastAPI server
    ├── agents/
    │   ├── travel_agent.py       # ReAct agent + self-critique
    │   ├── intent_classifier.py  # Structured LLM classifier
    │   ├── tools.py              # LangChain tool wrappers
    │   └── callback.py           # Observability callback
    ├── ml/
    │   ├── ranking.py            # LambdaMART LTR model
    │   └── scheduler.py          # DP itinerary scheduler
    ├── services/
    │   ├── geocoding.py          # Multi-source geocoding
    │   ├── places.py             # Google Places
    │   ├── routing.py            # Google Directions + TSP
    │   └── weather.py            # OpenWeatherMap
    ├── config/settings.py        # Centralised config
    └── tests/test_ml.py          # 25 unit tests
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Next.js 14, TypeScript, Tailwind CSS, Framer Motion |
| Backend | FastAPI, Uvicorn, Python 3.10+ |
| AI / Agents | LangChain ReAct, OpenAI GPT-4, LangSmith |
| ML | XGBoost (LambdaMART), NumPy |
| Maps | Google Maps, Google Directions, Google Places |
| Weather | OpenWeatherMap One Call API |
| Geocoding | Google Geocoding + Nominatim |
| Deployment | AWS EC2 (backend) + Vercel (frontend) |

---

## Running Locally

### Prerequisites
- Python 3.10+
- Node.js 18+
- API keys: OpenAI, Google Cloud, OpenWeatherMap

### Backend

```bash
cd backend
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
pip install -r requirements.txt
```

Create `.env` in the `backend/` folder:
```
OPENAI_API_KEY=sk-...
GOOGLE_API_KEY=AIza...
OPENWEATHER_API_KEY=...
LANGCHAIN_TRACING_V2=false
LANGCHAIN_API_KEY=
LANGCHAIN_PROJECT=travel-concierge
```

```bash
python -m uvicorn api:app --reload --port 8000
```

API docs → `http://localhost:8000/docs`

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Open → `http://localhost:3000`

### Tests

```bash
cd backend
pytest tests/ -v
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/chat` | Agent chat with intent routing |
| POST | `/api/itinerary` | DP-optimised multi-day schedule |
| POST | `/api/places` | LambdaMART re-ranked place search |
| POST | `/api/weather` | Current weather + forecast |
| POST | `/api/route` | TSP-optimised directions |
| POST | `/api/feedback` | Record accept/skip for LTR training |
| GET | `/api/health` | Health check |

---

## Google Cloud Setup

Enable these 3 APIs in [Google Cloud Console](https://console.cloud.google.com):
- Geocoding API
- Places API
- Directions API

One API key works for all three.

---

## Deployment

**Frontend → Vercel**
```bash
cd frontend
npx vercel
# Set env: NEXT_PUBLIC_API_URL=https://your-ec2-ip:8000
```

**Backend → AWS EC2**
```bash
git clone https://github.com/Satya-Chandana/Travel-Concierge.git
cd Travel-Concierge/backend
pip install -r requirements.txt
python -m uvicorn api:app --host 0.0.0.0 --port 8000
```
