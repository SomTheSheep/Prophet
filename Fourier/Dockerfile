FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy our custom script to the image
COPY fourier_simulator.py ./app.py

EXPOSE 8080

CMD ["python", "app.py"]