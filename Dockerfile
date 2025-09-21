# Multi-stage build for Node.js + Python
FROM node:18-bullseye

# Install Python and dependencies
RUN apt-get update && \
    apt-get install -y python3 python3-pip python3-dev && \
    ln -sf /usr/bin/python3 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# Set working directory
WORKDIR /app

# Copy and install frontend dependencies
COPY frontend/package*.json ./frontend/
RUN cd frontend && npm ci --only=production

# Copy and install backend dependencies
COPY backend/requirements.txt ./backend/
RUN cd backend && pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Build frontend
RUN cd frontend && npm run build

# Set environment variables
ENV PORT=8000
ENV PYTHONPATH=/app/backend

# Expose port
EXPOSE $PORT

# Start the application
CMD ["python", "main.py"]