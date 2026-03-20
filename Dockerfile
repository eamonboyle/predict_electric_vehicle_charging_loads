FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY configs ./configs
COPY datasets ./datasets

RUN pip install --no-cache-dir ".[api]"
RUN mkdir -p models artifacts runs

ENV EV_CHARGING_ROOT=/app
ENV EV_CHARGING_CONFIG=configs/default.yaml

EXPOSE 8000
CMD ["uvicorn", "ev_charging.api:app", "--host", "0.0.0.0", "--port", "8000"]
