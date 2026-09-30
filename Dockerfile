# 1. Use an official lightweight Python 3.12 image
FROM python:3.12-slim

# 2. Prevent Python from buffering stdout/stderr and writing .pyc files
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# 3. Set the working directory inside the container
WORKDIR /app

# 4. Install system packages (build tools needed for certain C-extensions like PyMuPDF)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 5. Copy requirements first to leverage Docker layer caching
COPY requirements.txt .

# 6. Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 7. Copy the rest of the application code
COPY . .

# 8. Expose Streamlit default port
EXPOSE 8501

# 9. Healthcheck to verify the web app is running
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# 10. Start the Streamlit application
ENTRYPOINT ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
