FROM python:3.11-slim-bookworm

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# create necessary directories
RUN mkdir -p data chroma_db

# Copy the rest of your application code
COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8502", "--server.address=0.0.0.0"]
