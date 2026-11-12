FROM python:3.12.3-slim-bookworm
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 libstdc++6 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN python -m pip install --no-cache-dir pip==24.0 && python -m pip install --no-cache-dir -r requirements.txt
COPY . .
RUN useradd --create-home --uid 1000 app && chown -R app:app /app
USER app
CMD ["uvicorn", "fastapi_app.app:app", "--host", "0.0.0.0", "--port", "8000"]
