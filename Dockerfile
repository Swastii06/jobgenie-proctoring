FROM python:3.11-slim

# Install system dependencies including cmake for dlib
RUN apt-get update && apt-get install -y \
    libpq-dev \
    gcc \
    g++ \
    cmake \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Run gunicorn
CMD ["gunicorn", "futurproctor.wsgi", "--bind", "0.0.0.0:8000", "--workers", "2", "--worker-class", "sync", "--timeout", "120", "--access-logfile", "-", "--error-logfile", "-"]
