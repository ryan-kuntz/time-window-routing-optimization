# Time-Window Routing Optimizer

**A personal routing optimization tool that solves the Orienteering Problem with Time Windows (OPTW), demonstrated through a real-world open house tour planning scenario.**

Given a set of candidate locations — each with an interest score, a visit window, and a visit duration — the optimizer selects which locations to visit and in what order to maximize total value within a time budget.

## Demo Use Case

The included notebook plans a Saturday open house tour across 17 home listings in the Lawrenceville/Lilburn/Duluth area (GA), priced between $350K–$500K. It walks through the full pipeline:

1. **Preference collection** — interactive widget captures buyer must-haves and feature weights
2. **Interest scoring** — weighted linear model ranks homes 0–100 based on price, size, schools, neighborhood vibe, etc.
3. **Geocoding** — three-tier fallback (US Census → Nominatim/OSM → trilateration from school locations)
4. **Travel time matrix** — OSRM road-network routing with haversine fallback
5. **OPTW solving** — brute force (exact) and greedy heuristic (baseline)

### Sample Result

With 7 eligible homes after filtering, the brute force solver finds a route visiting **all 7 homes** (score: 626.1) while the greedy heuristic manages only **4 homes** (score: 358.1, 57% of optimal). The greedy approach wastes ~3 hours waiting at its first stop — illustrating why routing order matters as much as selection.

## Project Structure

```
├── core/
│   ├── geocoding.py    # Three-tier geocoding pipeline + trilateration
│   ├── routing.py      # Duration matrix (OSRM + haversine fallback)
│   ├── scoring.py      # Weighted scoring model with dealbreaker filters
│   ├── solvers.py      # Brute force + greedy OPTW solvers
│   └── utils.py        # Time parsing utilities
├── data/
│   └── demo_homes.csv  # 17 sample home listings with full attributes
├── notebooks/
│   └── home_tour_exploration.ipynb  # End-to-end walkthrough
└── requirements.txt
```

## Getting Started

```bash
# Create and activate virtual environment
python -m venv venv
# Mac/Linux
source venv/bin/activate
# Windows
venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook notebooks/home_tour_exploration.ipynb
```

No API keys are required for now — geocoding and routing currently use free public services (US Census, Nominatim, OSRM).

## How It Works

### The OPTW Problem

The Orienteering Problem with Time Windows is NP-hard, combining:
- **Knapsack** — selecting which locations to include (each has a value and a time cost)
- **TSP** — determining the visit order to minimize travel time
- **Time windows** — each location can only be visited during its open window

### Scoring Model

A weighted linear scoring model (common in multi-criteria decision making) that:
- Applies hard dealbreaker filters first (min beds, max price, etc.)
- Normalizes and weights features (price, sqft, schools, neighborhood scores)
- Applies a property-type preference penalty
- Outputs a 0–100 interest score per home

### Geocoding Pipeline

A three-tier fallback handles the reality that no single free geocoder covers all addresses:
1. **US Census** — fast, reliable for established streets
2. **Nominatim/OSM** — catches newer developments Census misses
3. **Trilateration** — estimates coordinates from known school locations and Redfin-reported distances using scipy optimization

## Known Limitations

- **Brute force solver doesn't scale** — factorial complexity means it's only practical for ~10–12 homes
- **Geocoding depends on external services** — Census and Nominatim can be slow, rate-limited, or occasionally unavailable
- **Travel times are static** — OSRM gives point-to-point durations but doesn't account for time-of-day traffic
- **Trilateration accuracy** — the fallback geocoder estimates location from school distances, which can be off by ~0.5 miles depending on Redfin's rounding
- **Scoring model assumes feature independence** — price and sqft are correlated in reality but treated as separate axes
- **No persistent storage** — preferences and results live in notebook memory only; restarting the kernel requires re-running everything
