# 1. Use official lightweight Python 3.12 slim image
FROM python:3.12-slim

# 2. Prevent Python from buffering stdout/stderr and writing .pyc files
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# 3. Set the working directory
WORKDIR /app

# 4. Install temporary build dependencies and curl
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 5. CRITICAL: Install CPU-only PyTorch first (saves ~7-8 GB)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# 6. Copy requirements and install the remaining dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 7. Clean up build tools to slim down the image further (~300-400 MB saved)
RUN apt-get purge -y --auto-remove build-essential \
    && rm -rf /var/lib/apt/lists/*

# 8. Copy application code
COPY . .

# 9. Expose Streamlit default port
EXPOSE 8501

# 10. Healthcheck
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# 11. Start Streamlit
ENTRYPOINT ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
