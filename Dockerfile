FROM python:3.10-slim-bullseye

# Install TA-Lib dependencies
RUN apt-get update && \
    apt-get install -y build-essential gcc && \
    apt-get clean

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD gunicorn -w 4 -b 0.0.0.0:$PORT app:app
