FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libpq-dev \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Collect static files
RUN python manage.py collectstatic --noinput || true

# Run gunicorn
CMD ["gunicorn", "futurproctor.wsgi", "--bind", "0.0.0.0:8000", "--workers", "2", "--timeout", "120"]
