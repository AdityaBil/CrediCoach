# CrediCoach — Local Web UI

This adds a minimal Flask backend and a static HTML/CSS/JS frontend for the existing Python models.

Run locally:

```bash
python -m pip install -r requirements.txt
python server.py
```

Then open http://localhost:5000 in your browser. The server trains models on startup (may take a minute) and serves the frontend from `/static`.

Docker
------

Build and run with Docker (creates a quicker dev container using `FAST_MODE=1`):

```bash
docker build -t credicoach .
docker run -p 5000:5000 -e FAST_MODE=1 credicoach
```

Or with docker-compose:

```bash
docker-compose up --build
```

Notes:
- `FAST_MODE=1` reduces model training (fewer trees/epochs) so the container starts faster. Set to `0` to run full training.
 
Gemini (optional)
------------------

If you have access to Gemini via the `genai` package, set `GOOGLE_API_KEY` (or `GENAI_API_KEY`) in your environment before starting the server. When present, the server will call Gemini to generate the personalized advice instead of using the local fallback.

Example:

```bash
export GOOGLE_API_KEY="ya29.your_key_here"
python server.py
```

