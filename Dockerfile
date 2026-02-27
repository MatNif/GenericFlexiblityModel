# docker build -t flexmodel -f Dockerfile .
FROM python:3.11-slim

# Prevent Python from writing .pyc files
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Copy project
COPY . /app

# Install system deps (scipy needs build tools sometimes)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install project + dashboard deps
RUN pip install --upgrade pip \
    && pip install . \
    && pip install streamlit plotly pandas scipy numpy

EXPOSE 8501

CMD ["streamlit", "run", \
     "examples/battery_vs_market/dashboard.py", \
     "--server.address=0.0.0.0", \
     "--server.port=8501", \
     "--server.headless=true", \
     "--server.baseUrlPath=flexmodel"]