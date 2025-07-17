# Backend

This directory contains the Docker configuration and application code for the GCT SaaS API service.

- **app/** - FastAPI application source code.
- **requirements.txt** and **setup.py** - Python dependencies.
- **sql/** - database initialization scripts.

Start the API locally from the repository root:

```bash
docker-compose up api
```

The service will be available at `http://localhost:8000`.
