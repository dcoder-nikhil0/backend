# Use official Python slim image as base
FROM python:3.9-slim

# Set working directory inside container
WORKDIR /server

# Copy requirements.txt and install dependencies
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files to container
COPY . .

# Expose Flask port
EXPOSE 5000

# Run the app with host=0.0.0.0 for external access
CMD ["python", "app.py"]
