# FastAPI Application

This folder contains the implementation of the GCT SaaS API built with FastAPI.

Subdirectories:
- **api/** – versioned API routers (currently `v1`).
- **core/** – coherence formulas and derivative calculators.
- **models/** – dataclasses for persistence.
- **services/** – integrations like `GrokService` and the derivative-based
  `LeadershipEvaluator` for leadership analysis over time.

The entry point of the service is `main.py`.
