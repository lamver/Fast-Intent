#!/bin/bash
set -e

MODELS_DIR="./models"
mkdir -p $MODELS_DIR

# Список моделей для скачивания (код_языка)
LANGS=("ru" "en" "es")

# 1. Скачиваем LID модель (определитель языка)
if [ ! -f "${MODELS_DIR}/lid.176.bin" ]; then
    echo "Downloading language ID model..."
    curl -L https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin -o "${MODELS_DIR}/lid.176.bin"
fi

# 2. Скачиваем языковые векторы
for lang in "${LANGS[@]}"; do
    FILE="${MODELS_DIR}/cc.${lang}.300.bin"
    if [ ! -f "$FILE" ]; then
        echo "Downloading $lang model..."
        curl -L "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.${lang}.300.bin.gz" | gunzip > "$FILE"

    else
        echo "$lang model already exists."
    fi
done

echo "All models are ready!"