# Dockerfile — Builds the LLM Output Quality Reviewer environment container.
#
# Build :  docker build -t llm-quality-reviewer .
# Run   :  docker run -p 7860:7860 llm-quality-reviewer
#
# HuggingFace Spaces requires port 7860 and a non-root user.

FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app

# Install Python dependencies first (cached layer — only rebuilds if requirements change)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the source code
COPY . .

# HuggingFace Spaces runs containers as a non-root user for security
RUN useradd -m appuser
USER appuser

# Expose the port HuggingFace expects
EXPOSE 7860

# Start the FastAPI server on all interfaces at port 7860
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]