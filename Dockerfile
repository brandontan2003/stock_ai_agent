FROM python:3.11-slim

WORKDIR /app
COPY .. /app

RUN pip install --upgrade pip && pip install --no-cache-dir --upgrade -r requirements.txt

ENV PYTHONPATH /app
EXPOSE 8080

ENTRYPOINT ["python", "-u", "main.py"]