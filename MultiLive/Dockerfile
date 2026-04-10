FROM python:3.10-slim

WORKDIR /app

# Ensure you run the build command from the ROOT of the Prophet-main directory
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Pre-compile cmdstan (required for Prophet)
RUN python -c "from prophet import Prophet; Prophet()"

COPY MultiLive/multilive.py ./app.py

EXPOSE 8080

CMD ["python", "app.py"]