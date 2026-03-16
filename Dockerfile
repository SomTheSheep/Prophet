FROM python:3.10-slim

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-compile cmdstan (required for Prophet)
RUN python -c "from prophet import Prophet; Prophet()"

# Copy application
COPY app.py .
COPY config.yaml .

EXPOSE 8000

CMD ["python", "app.py"]
