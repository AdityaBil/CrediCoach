FROM python:3.11-slim

WORKDIR /app

# system deps for some packages
RUN apt-get update && apt-get install -y build-essential gcc git --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN python -m pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# default to fast mode for quicker container startup; override with env var
ENV FAST_MODE=1

COPY . .

EXPOSE 5000
CMD ["python", "server.py"]
