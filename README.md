# Ollama Chatbot

A FastAPI-based chatbot application that integrates with Ollama for local LLM inference, featuring health monitoring, query processing, and embedding capabilities.

## Features

- **Local LLM Integration**: Uses Ollama for running language models locally
- **FastAPI Backend**: RESTful API with automatic documentation
- **Health Monitoring**: Built-in health check endpoints
- **Query Processing**: Advanced query handling and response generation
- **Embedding Support**: Text embeddings using specialized models
- **MongoDB Integration**: Data persistence and storage
- **Docker Support**: Containerized deployment for easy setup

## Project Structure

```
ollama-chatbot/
├── src/
│   ├── models/
│   │   ├── __init__.py
│   │   ├── database_model.py
│   │   ├── health_model.py
│   │   ├── query_model.py
│   │   └── response_model.py
│   ├── __init__.py
│   ├── chat.py
│   ├── config.py
│   ├── dataset.py
│   ├── embeddings.py
│   ├── main.py
│   ├── mongodb.py
│   └── utils.py
├── .env
├── .env.example
├── .dockerignore
├── .gitignore
├── docker-compose.yml
├── Dockerfile
├── entrypoint.sh
├── LICENSE
├── README.md
└── requirements.txt
```

## Prerequisites

- **Docker & Docker Compose**: For containerized deployment
- **Python 3.8+**: For local development
- **MongoDB**: Database (can be run via Docker)
- **Git**: For version control

## Installation & Setup

### Method 1: Docker Compose (Recommended)

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd ollama-chatbot
   ```

2. **Set up environment variables**:
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

3. **Build and run with Docker Compose**:
   ```bash
   docker-compose up --build
   ```

4. **Wait for initialization**:
   The container will automatically:
   - Start the Ollama server
   - Download required models (llama3, embeddings models)
   - Launch the FastAPI application

5. **Access the application**:
   - FastAPI app: http://localhost:3000
   - API documentation: http://localhost:3000/docs
   - Ollama server: http://localhost:11434

### Method 2: Local Development

1. **Install Ollama**:
   ```bash
   curl https://ollama.com/install.sh | sh
   ```

2. **Start Ollama server**:
   ```bash
   ollama serve
   ```

3. **Pull required models**:
   ```bash
   ollama pull llama3
   ollama pull hf.co/CompendiumLabs/bge-base-en-v1.5-gguf
   ollama pull hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF
   ```

4. **Set up Python environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

5. **Configure environment**:
   ```bash
   cp .env.example .env
   # Edit .env with your settings
   ```

6. **Run the application**:
   ```bash
   export PYTHONPATH=./src
   uvicorn src.main:app --host 0.0.0.0 --port 3000 --reload
   ```

## Environment Configuration

Create a `.env` file based on `.env.example`:

```env
# MongoDB Configuration
MONGODB_URL=mongodb://localhost:27017
MONGODB_DATABASE=ollama_chatbot

# Ollama Configuration
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3

# Application Configuration
DEBUG=true
LOG_LEVEL=info
```

## API Endpoints

### Health Check
- `GET /health` - Application health status
- `GET /health/detailed` - Detailed health information

### Chat
- `POST /chat` - Send chat messages
- `GET /chat/history` - Retrieve chat history

### Models
- `GET /models` - List available Ollama models
- `POST /models/pull` - Pull new models

### Embeddings
- `POST /embeddings` - Generate text embeddings

## Development

### Hot Reload
When running locally, the application supports hot reload for development:
```bash
uvicorn src.main:app --host 0.0.0.0 --port 3000 --reload
```

### Docker Development
For development with Docker, the `docker-compose.yml` mounts your source code:
```bash
docker-compose up --build
# Make changes to src/ files - they'll be reflected immediately
```

### Adding New Models
To add new Ollama models:
1. Update `entrypoint.sh` to pull the model
2. Configure the model in your `.env` file
3. Restart the container

## Troubleshooting

### Common Issues

1. **Ollama connection refused**:
   - Ensure Ollama server is running on port 11434
   - Check firewall settings

2. **Models not downloading**:
   - Verify internet connection
   - Check Ollama server logs
   - Ensure sufficient disk space

3. **MongoDB connection errors**:
   - Verify MongoDB is running
   - Check connection string in `.env`

4. **Port conflicts**:
   - Change ports in `docker-compose.yml` if needed
   - Ensure ports 3000 and 11434 are available

### Logs
View application logs:
```bash
# Docker Compose
docker-compose logs -f

# Individual container
docker logs <container-name>
```