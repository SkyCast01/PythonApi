# # -------------------------------------------------------
# # Builder: install Python deps into /root/.local (cacheable)
# # -------------------------------------------------------
# FROM python:3.10-slim AS builder

# # System build deps (kept only in builder)
# RUN apt-get update && apt-get install -y --no-install-recommends \
#         build-essential gcc g++ \
#     && rm -rf /var/lib/apt/lists/*

# WORKDIR /app

# # Copy only requirements first to leverage Docker layer caching
# COPY medical_service/requirements.txt ./requirements.txt

# # Install Python deps to user site (goes under /root/.local)
# RUN python -m pip install --upgrade pip setuptools wheel && \
#     pip install --user --no-cache-dir -r requirements.txt

# # -------------------------------------------------------
# # Runtime: slim image + only the libs needed to run
# # -------------------------------------------------------
# FROM python:3.10-slim

# ENV PYTHONDONTWRITEBYTECODE=1 \
#     PYTHONUNBUFFERED=1 \
#     MODEL_DIR="/app/AiModels" \
#     PATH="/root/.local/bin:$PATH"

# # Lightweight runtime OS libs commonly needed by numpy/scikit-learn/torch/Pillow
# RUN apt-get update && apt-get install -y --no-install-recommends \
#         libstdc++6 \
#         libgomp1 \
#         libglib2.0-0 \
#         libgl1 \
#         libsm6 \
#         libxext6 \
#         libxrender1 \
#     && rm -rf /var/lib/apt/lists/*

# WORKDIR /app

# # Bring in the installed Python packages from builder
# COPY --from=builder /root/.local /root/.local

# # Copy application code
# COPY medical_service/ ./medical_service/
# COPY AiModels/ /app/AiModels/

# # (Optional) Pre-download NLTK data at build time to speed up cold starts
# # Uncomment if you prefer not to download at runtime
# # RUN python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('stopwords', quiet=True)"

# # Expose API port
# EXPOSE 6000

# # Start the app
# CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "6000"]

# Use a lightweight Python image
# Use a lightweight Python image
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY . .

# Copy models into container
COPY models/ ./models/

# Expose FastAPI port
EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "2000"]
