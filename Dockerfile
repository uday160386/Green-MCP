# syntax=docker/dockerfile:1
FROM python:3.11-slim

# Set environment variables for Green AI
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PDF_CACHE_SIZE=2
ENV PDF_CACHE_TIMEOUT=600
ENV WEATHER_THROTTLE_SECONDS=10

WORKDIR /app


# Install system dependencies
RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY . /app


# Install Python dependencies and uv
RUN pip install --upgrade pip && \
    pip install --no-cache-dir uv && \
    pip install --no-cache-dir -r requirements.txt || pip install --no-cache-dir -r pyproject.toml

# Expose port (if using HTTP transport)
EXPOSE 8000

# Run the MCP server
CMD ["uv", "run", "./server/green-mcp-server.py"]
