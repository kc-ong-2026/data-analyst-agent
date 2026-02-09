# Govtech Chat Assistant

An AI-powered chat assistant for analyzing Singapore government data with real-time visualization capabilities.

## Features

- **Chat Interface**: Interactive chat with AI for data analysis queries
- **Data Visualization**: Dynamic charts (bar, line, pie, scatter) rendered on the right panel
- **Model Agnostic**: Support for multiple LLM providers (OpenAI, Anthropic, Google)
- **Configurable**: YAML-based configuration with environment variable support
- **Docker Ready**: Full Docker and Docker Compose setup for easy deployment

## Architecture

```
├── frontend/          # React + TypeScript + Vite
│   ├── src/
│   │   ├── components/   # React components
│   │   ├── store/        # Zustand state management
│   │   ├── api/          # API client
│   │   └── types/        # TypeScript types
│   └── Dockerfile
│
├── backend/           # Python FastAPI
│   ├── app/
│   │   ├── routes/       # API endpoints
│   │   ├── services/     # Business logic
│   │   └── config.py     # Configuration
│   ├── config/
│   │   └── config.yaml   # YAML configuration
│   └── Dockerfile
│
├── dataset/           # Singapore manpower datasets
├── api_spec/          # API specifications
└── docker-compose.yml
```

## Quick Start

### Prerequisites

- Docker and Docker Compose
- API keys for at least one LLM provider (OpenAI, Anthropic, or Google)

### Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Govtech_assignment
   ```

2. **Configure environment variables**
   ```bash
   cp backend/.env.example backend/.env
   ```

   Edit `backend/.env` and add your API keys:
   ```
   OPENAI_API_KEY=your_key_here
   ANTHROPIC_API_KEY=your_key_here
   GOOGLE_API_KEY=your_key_here
   ```

3. **Start with Docker Compose**
   ```bash
   docker-compose up --build
   ```

4. **Access the application**
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:8000
   - API Docs: http://localhost:8000/docs

### Development Mode

For development with hot reload:

```bash
docker-compose -f docker-compose.dev.yml up --build
```

Or run services locally:

```bash
# Backend
cd backend
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows
pip install -r requirements.txt
cp .env.example .env  # and add your API keys
uvicorn app.main:app --reload

# Frontend (new terminal)
cd frontend
npm install
npm run dev
```

## Configuration

### Backend Configuration (`backend/config/config.yaml`)

```yaml
llm:
  providers:
    openai:
      models:
        - gpt-4-turbo-preview
        - gpt-3.5-turbo
      default_model: gpt-4-turbo-preview
      temperature: 0.7
      max_tokens: 4096
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key | - |
| `ANTHROPIC_API_KEY` | Anthropic API key | - |
| `GOOGLE_API_KEY` | Google API key | - |
| `DEFAULT_LLM_PROVIDER` | Default LLM provider | openai |
| `DEFAULT_LLM_MODEL` | Default model | gpt-4-turbo-preview |
| `HOST` | Backend host | 0.0.0.0 |
| `PORT` | Backend port | 8000 |

## API Endpoints

### Chat
- `POST /api/chat/` - Send a message and get response with visualization
- `GET /api/chat/history/{conversation_id}` - Get conversation history
- `DELETE /api/chat/history/{conversation_id}` - Clear conversation

### Configuration
- `GET /api/config/` - Get current configuration
- `GET /api/config/providers` - Get available providers and models
- `GET /api/config/health` - Health check

### Data
- `GET /api/data/datasets` - List available datasets
- `GET /api/data/datasets/{path}/info` - Get dataset information
- `GET /api/data/datasets/{path}/query` - Query dataset

## Tech Stack

### Frontend
- React 18
- TypeScript
- Vite
- Tailwind CSS
- Recharts (visualization)
- Zustand (state management)

### Backend
- Python 3.11
- FastAPI
- LangChain & LangGraph
- Pydantic

## License

MIT
