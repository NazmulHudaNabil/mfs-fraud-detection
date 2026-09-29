# Use a minimal Python image to keep the container lightweight
FROM python:3.12-slim-bookworm

# Security best practice: Create a non-root user early before copying files
RUN useradd -m appuser

# Set the working directory inside the container
WORKDIR /app

# Set environment variables for better Python execution
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="/app"

# Copy the requirements file first to leverage Docker cache
# Use --chown to prevent duplicating file sizes in Docker layers
COPY --chown=appuser:appuser requirements.txt .

# Install dependencies without keeping the cache to save space
RUN pip install --no-cache-dir -r requirements.txt

# Copy the necessary application code and model artifacts
# Using --chown here prevents the 500MB+ image bloat!
COPY --chown=appuser:appuser src/ /app/src/
COPY --chown=appuser:appuser artifacts/ /app/artifacts/

# Switch to the non-root user
USER appuser

# Expose a default port for local testing
EXPOSE 8000

# Command to run the FastAPI + Gradio app
# We use JSON array format with 'exec' so OS signals (like stopping the container) work correctly,
# while still allowing the shell to inject Heroku's dynamic $PORT.
CMD ["sh", "-c", "exec uvicorn src.app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
