# [DOCKER DESKTOP]
# This Dockerfile is compatible with Docker Desktop as-is.
# To build locally:
#   docker build -t youtube-comment-analyzer .
# To run locally:
#   docker run -p 5000:5000 youtube-comment-analyzer

FROM python:3.11-slim-buster

WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install dependencies first for caching of layers
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Expose port
EXPOSE 5000

# Run the application
CMD ["python", "flask_api/main.py"]