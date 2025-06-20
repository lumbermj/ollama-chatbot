# Step 1: Use an official Python runtime as a parent image
FROM python:3.9-slim

# Step 2: Set the working directory in the container
WORKDIR /app

# Step 3: Copy the current directory contents into the container at /app
COPY . /app

# Step 4: Install the necessary dependencies
RUN pip install --no-cache-dir -r requirements.txt

COPY ./src /app/src

# Set the working directory to the src folder
WORKDIR /app/src

# Step 5: Set environment variables from the .env file (optional, can be done via docker-compose or environment)
ENV PYTHONUNBUFFERED=1

# Step 6: Expose port 5000 if you're running a web app (optional, adjust based on your app's requirements)
EXPOSE 5000

# Step 7: Define the command to run the application
CMD ["python", "main.py"]