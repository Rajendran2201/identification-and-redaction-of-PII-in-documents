# Use Node.js base image with Python support
FROM node:18-bullseye

# Install Python and pip
RUN apt-get update && \
    apt-get install -y python3 python3-pip python3-venv && \
    ln -s /usr/bin/python3 /usr/bin/python

# Set working directory
WORKDIR /app

# Copy package files
COPY frontend/package*.json ./frontend/
COPY backend/requirements.txt ./backend/

# Install frontend dependencies
RUN cd frontend && npm install

# Install backend dependencies
RUN cd backend && pip3 install -r requirements.txt

# Copy application code
COPY . .

# Build frontend
RUN cd frontend && npm run build

# Expose port
EXPOSE $PORT

# Start command
CMD cd backend && python -m uvicorn main:app --host 0.0.0.0 --port $PORT