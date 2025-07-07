FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code
COPY . .

# create necessary directories
RUN mkdir -p data chroma_db 

EXPOSE 8501

CMD ["streamlit", "run", "app.py"]