# Fast-Intent

Fast-Intent is a production-ready NLP microservice built with FastAPI and FastText. It provides language detection, semantic similarity comparison, and intent routing using pre-trained FastText models. The service is containerized with Docker and can be easily deployed with Docker Compose.

## Live Demo

You can test the service interactively at **[datahunter.store/tools/text/compare](datahunter.store/tools/text/compare)**.

## Features

- **Language Detection** – Detect the language of a text snippet using FastText's LID (Language Identification) model supporting 176 languages.
- **Semantic Similarity** – Compute cosine similarity between two text snippets using FastText word embeddings (available for Russian, English, and Spanish).
- **Intent Routing** – Route user text to the most relevant intent based on example phrases (multi‑class classification).
- **IP Whitelist & Dynamic IP Updates** – Restrict access via static allowed IPs and/or a remotely fetched list (updated every 10 minutes).
- **Health & Debug Endpoints** – Check service status, inspect loaded models, and manually trigger IP list refresh.
- **Automatic Model Download** – Models are downloaded automatically via a separate Docker Compose service (`downloader`).
- **Scalable** – Runs with Gunicorn and Uvicorn workers; the number of workers can be configured via environment variables.

## Quick Start

### Prerequisites

- Docker and Docker Compose installed on your system.
- Git (optional, for cloning the repository).

### Using Docker Compose (Recommended)

1. Clone the repository (or download the source code):
   ```bash
   git clone <repository-url>
   cd fast-intent
   ```

2. Copy the example environment file and adjust if needed:
   ```bash
   cp .env.example .env
   ```

3. Start the service:
   ```bash
   docker-compose up --build
   ```

   The first run will download the required FastText models (approx. 6‑7 GB) into the `./models` directory. This may take several minutes depending on your internet connection.

4. Once the services are up, the API will be available at `http://localhost:8000`.

### Running Without Docker

If you prefer to run the service directly on your host:

1. Install Python 3.10 (or later) and pip.

2. Install the Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the required models manually (or run the `prepare_models.sh` script if available). The models must be placed in a `models/` directory relative to the application:
   - `lid.176.bin` – language identification model.
   - `cc.ru.300.bin`, `cc.en.300.bin`, `cc.es.300.bin` – FastText word‑embedding models for Russian, English, and Spanish.

4. Start the FastAPI server with Uvicorn:
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   ```

## API Endpoints

All endpoints accept and return JSON.

### `POST /detect-language`

Detects the language of the provided text and returns the top‑3 predictions with confidence scores.

**Request:**
```json
{
  "text": "Your input text here"
}
```

**Response:**
```json
{
  "text": "Your input text here",
  "predictions": [
    {"language": "en", "confidence": 0.95},
    {"language": "de", "confidence": 0.03},
    {"language": "fr", "confidence": 0.02}
  ]
}
```

### `POST /compare-vectors`

Compares two text snippets and returns their semantic similarity (cosine similarity) based on FastText sentence vectors. The language is automatically detected from the first text.

**Request:**
```json
{
  "text1": "First piece of text",
  "text2": "Second piece of text"
}
```

**Response:**
```json
{
  "detected_language": "en",
  "similarity": 0.8732,
  "percentage": "87.32%"
}
```

### `GET /healthcheck`

Returns the health status of the service and the number of loaded vector models.

**Response:**
```json
{
  "status": "ok",
  "models_count": 3
}
```

### `GET /debug-models`

Lists all model files present in the `/app/models` directory along with their sizes and modification timestamps. Useful for verifying that models have been downloaded correctly.

**Response:**
```json
{
  "files_info": [
    {
      "file": "lid.176.bin",
      "size_mb": 131.5,
      "modified": "2026-04-20 10:30:45"
    },
    ...
  ],
  "loaded_keys_in_dict": ["ru", "en", "es"],
  "lang_model_loaded": true,
  "current_working_dir": "/app"
}
```

### `POST /route-intent`

Routes a user text to the most relevant intent based on provided example phrases for each intent.

**Request:**
```json
{
  "text": "I want to listen to rock music",
  "intents": {
    "music": ["play a song", "music playlist", "rock music"],
    "movies": ["watch a film", "movie recommendations", "horror movies"],
    "weather": ["what's the weather", "is it raining"]
  }
}
```

**Response:**
```json
{
  "text": "I want to listen to rock music",
  "intent": "music",
  "confidence": 0.87
}
```
If no intent reaches the confidence threshold (default 0.4), the service returns `"intent": "fallback"`.

### `GET /refresh-ips`

Manually triggers an update of the dynamic IP whitelist from the remote URL. Requires a `token` query parameter that matches the `REFRESH_TOKEN` environment variable.

**Request:**
```
GET /refresh-ips?token=your-secret-token
```

**Response (success):**
```json
{
  "status": "success",
  "updated_ips": ["203.0.113.5", "198.51.100.10"]
}
```

## Configuration

Environment variables are read from a `.env` file (see `.env.example` for the template).

| Variable | Description | Default |
|----------|-------------|---------|
| `PORT`   | Port on which the FastAPI application listens. | `8000` |
| `WORKERS`| Number of Gunicorn worker processes. | `4` |
| `TIMEOUT`| Request timeout in seconds (used by Gunicorn). | `120` |
| `ALLOWED_IPS` | Comma‑separated list of IP addresses allowed to access the API (excluding `/healthcheck` and `/debug‑models`). | `127.0.0.1` |
| `REMOTE_IPS_URL` | URL that returns a comma‑ or newline‑separated list of IP addresses; the service fetches and updates this list every 10 minutes. | – |
| `REFRESH_TOKEN` | Secret token required for the manual IP‑refresh endpoint (`/refresh‑ips`). | – |

You can also adjust resource limits (CPU, memory) in the `docker-compose.yaml` file under the `deploy.resources.limits` section.

## Project Structure

```
fast-intent/
├── main.py                 # FastAPI application (endpoints, model loading)
├── dockerfile              # Docker image definition
├── docker-compose.yaml     # Docker Compose configuration (includes downloader service)
├── requirements.txt        # Python dependencies
├── .env.example            # Example environment variables
├── models/                 # Directory for downloaded FastText models (auto‑created)
└── README.md               # This file
```

## How It Works

1. **Model Loading** – At startup, the service loads the language‑detection model (`lid.176.bin`) and the word‑embedding models for Russian, English, and Spanish (if they exist in `/app/models`). The models are loaded once and reused across all requests.

2. **Language Detection** – The `lid.176.bin` model predicts the language of a given text. It returns up to three most probable language codes with confidence scores.

3. **Sentence Vectors** – For similarity comparison, the service uses FastText’s `get_sentence_vector` method, which averages word vectors to produce a fixed‑length representation of the input text.

4. **Similarity Calculation** – Cosine similarity is computed between the two sentence vectors using `scikit‑learn`’s `cosine_similarity` function.

5. **Intent Routing** – For the `/route‑intent` endpoint, the service computes the vector of the input text and compares it with vectors of example phrases for each intent. The intent with the highest cosine similarity above a threshold (0.4) is selected.

6. **IP Whitelisting** – Every request (except `/healthcheck` and `/debug‑models`) passes through a middleware that checks the client IP against a static list (`ALLOWED_IPS`) and a dynamically updated list fetched from `REMOTE_IPS_URL`. The dynamic list is refreshed every 10 minutes in a background task, and can also be updated manually via `/refresh‑ips` (requires a token).

## Troubleshooting

- **Models not downloading** – Check the logs of the `downloader` service. Ensure you have enough disk space (≈7 GB) and a stable internet connection. You can also download the models manually and place them in the `models/` folder.

- **High memory usage** – The FastText models are memory‑intensive. If you run out of memory, reduce the number of workers (`WORKERS` environment variable) or limit the container’s memory in `docker-compose.yaml`.

- **Slow first request** – The first request after startup may be slower because models are loaded lazily. Subsequent requests are fast.

- **Language detection fails** – If the language detection model is not loaded, the service falls back to Russian (`"ru"`). Verify that `lid.176.bin` exists in the `models/` directory.

## License

[Specify your license here, e.g., MIT]

## Acknowledgments

- Built with [FastAPI](https://fastapi.tiangolo.com/) and [Uvicorn](https://www.uvicorn.org/).
- Uses [FastText](https://fasttext.cc/) for language identification and word embeddings.
- Model files are provided by [Facebook Research](https://fasttext.cc/docs/en/language-identification.html).

---
*For questions or contributions, please open an issue or submit a pull request.*
