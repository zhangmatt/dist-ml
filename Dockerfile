FROM python:3.12-slim

RUN apt-get update \
  && apt-get install -y --no-install-recommends cmake g++ make \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN cmake -S . -B build && cmake --build build

CMD ["python3", "scripts/train_distributed.py", "--param-server-bin", "/app/build/param_server", "--output", "/models/model.weights"]
