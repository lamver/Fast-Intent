FROM python:3.10-slim

# Устанавливаем системные зависимости, необходимые для Piper и ONNX
RUN apt-get update && apt-get install -y \
    libasound2 \
    libsndfile1 \
    curl \
    gzip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Скачиваем модель для ОПРЕДЕЛЕНИЯ языка (176 языков в одном файле)
#RUN curl -L https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin -o lid.176.bin
#RUN curl -L https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.bin.gz | gunzip > cc.en.300.bin
#RUN curl -L https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.es.300.bin.gz | gunzip > cc.es.300.bin
#RUN curl -L https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.ru.300.bin.gz | gunzip > cc.ru.300.bin


# Скачиваем модель для ВЕКТОРОВ (например, русскую)
#RUN curl -L https://fbaipublicfiles.com | gunzip > cc.ru.300.bin

COPY . .

# Делаем скрипт исполняемым внутри образа
#RUN chmod +x download_models.sh

# Скрипт будет запускаться первым, а затем вызывать CMD
#ENTRYPOINT ["./download_models.sh"]

# Проверьте, что в .env WORKERS=48, а не больше, чем ядер
CMD gunicorn main:app \
    -w ${WORKERS:-4} \
    -k uvicorn.workers.UvicornWorker \
    -b 0.0.0.0:${PORT:-8000} \
    --timeout ${TIMEOUT:-120} \
    --preload