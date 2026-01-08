# How to Run CrediCoach

## Prerequisites

- **Node.js 18+** and npm
- **Python 3.11+** and pip

## Step 1: Install Dependencies

### Frontend (Node.js)
```bash
npm install
```

### Backend (Python)
```bash
pip install -r requirements.txt
```

## Step 2: Run the Application

You need to run **both** the backend and frontend servers.

### Terminal 1: Start the Backend Server

```bash
python server.py
```

The backend will:
- Load or train ML models (first time may take a few minutes)
- Start on `http://localhost:5000`
- API endpoints available at:
  - `POST http://localhost:5000/api/predict` - Get loan predictions
  - `GET http://localhost:5000/api/history` - Get prediction history

**Note:** On first run, if models don't exist, it will train them which may take several minutes.

### Terminal 2: Start the Frontend Development Server

```bash
npm run dev
```

The frontend will:
- Start on `http://localhost:5173` (or next available port)
- Automatically open in your browser
- Hot-reload on code changes

## Access the Application

Open your browser and go to: **http://localhost:5173**

---

## Optional: Environment Variables

You can set these environment variables for the backend:

- `FAST_MODE=1` - Use faster model training (fewer estimators)
- `LOAD_MODELS=0` - Force retraining of models
- `GOOGLE_API_KEY` or `GENAI_API_KEY` - Enable Gemini AI for enhanced financial advice

Example:
```bash
# Windows PowerShell
$env:FAST_MODE="1"
python server.py

# Windows CMD
set FAST_MODE=1
python server.py

# Linux/Mac
export FAST_MODE=1
python server.py
```

---

## Production Build

### Build Frontend
```bash
npm run build
```

This creates a `dist/` folder with production-ready files.

### Run Production Build
The Flask server can serve the built frontend. Update `server.py` to serve from `dist/` folder.

---

## Docker Deployment

### Using Docker Compose
```bash
docker-compose up --build
```

### Using Docker
```bash
docker build -t credicoach .
docker run -p 5000:5000 credicoach
```

---

## Troubleshooting

### Backend Issues
- **Models not loading**: Ensure `focused_synthetic_loan_data.csv` exists in the root directory
- **Port 5000 already in use**: Change the port in `server.py` (line 304)
- **Missing dependencies**: Run `pip install -r requirements.txt`

### Frontend Issues
- **Port 5173 already in use**: Vite will automatically use the next available port
- **Build errors**: Try deleting `node_modules` and running `npm install` again

### Model Training
- First run will train models (can take 5-10 minutes)
- Models are saved in `models/` directory for faster subsequent starts
- Set `FAST_MODE=1` for quicker training during development

