# Base image for Ollama & Python
FROM ubuntu:22.04

# Install system dependencies
RUN apt-get update && \
    apt-get install -y curl python3 python3-venv python3-pip ca-certificates && \
    update-ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Install Ollama CLI
RUN curl https://ollama.com/install.sh | sh

# Install Ollama using direct binary download (more reliable in Docker)
#RUN curl -L -o ollama https://github.com/ollama/ollama/releases/latest/download/ollama-linux-amd64 && \
#    chmod +x ollama && \
#    mv ollama /usr/local/bin/ollama

# Set working directory
WORKDIR /app

# Copy dependency list and install Python virtualenv
COPY requirements.txt .
RUN python3 -m venv venv && \
    venv/bin/pip install --upgrade pip && \
    venv/bin/pip install -r requirements.txt

# Copy source code into the container
COPY src ./src

# Copy and make the entrypoint script executable
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Expose Ollama and FastAPI ports
EXPOSE 11434 3000

# Use the entrypoint to start services
ENTRYPOINT ["/entrypoint.sh"]