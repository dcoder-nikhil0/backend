# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Copy requirement files and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the full project into the container
COPY . .

# Change to the 'server' directory where app.py is located
WORKDIR /app/server

# Run the app
CMD ["python", "app.py"]
