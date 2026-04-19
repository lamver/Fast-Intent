import os
import fasttext
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from typing import List

# Создаем экземпляр приложения
app = FastAPI(title="NLP Service")

# Глобальные переменные для моделей
vector_model = None
lang_model = None
vector_models = {}

@app.on_event("startup")
def load_models():
    global lang_model, vector_models
    try:
        # Загружаем определитель
        lang_model = fasttext.load_model("/app/models/lid.176.bin")
        
        # Загружаем векторы в словарь
        for lang in ["ru", "en", "es"]:
            path = f"/app/models/cc.{lang}.300.bin"
            if os.path.exists(path):
                vector_models[lang] = fasttext.load_model(path)
                print(f"Loaded {lang} model")
    except Exception as e:
        print(f"Startup error: {e}")
        
class TextRequest(BaseModel):
    text: str

@app.post("/detect-language", tags=["NLP"])
async def detect_language(data: TextRequest):
    if lang_model is None:
        raise HTTPException(status_code=503, detail="Language model not loaded")
    
    # Берем топ-5, чтобы было из чего фильтровать
    labels, probabilities = lang_model.predict(data.text, k=5)
    
    # Фильтруем: оставляем только те, где уверенность > 0.01
    results = [
        {
            "language": label.replace("__label__", ""),
            "confidence": round(float(prob), 4)
        }
        for label, prob in zip(labels, probabilities)
        if float(prob) >= 0.01
    ]
    
    return {
        "text": data.text,
        "top_predictions": results
    }
    
class CompareRequest(BaseModel):
    text1: str
    text2: str

@app.post("/compare-vectors", tags=["NLP"])
async def compare_texts(data: CompareRequest):
    global vector_models, lang_model
    # 1. Определяем язык первого текста, чтобы выбрать модель
    labels, _ = lang_model.predict(data.text1, k=1)
    lang = labels[0].replace("__label__", "")
    
    # 2. Берем нужную модель (например, русскую по дефолту, если язык не поддерживается)
    model = vector_models.get(lang, vector_models.get("ru"))
    
    if not model:
        raise HTTPException(status_code=503, detail="Suitable vector model not loaded")
    
    # 3. Получаем векторы
    v1 = model.get_sentence_vector(data.text1).reshape(1, -1)
    v2 = model.get_sentence_vector(data.text2).reshape(1, -1)
    
    # 4. Считаем сходство
    similarity = cosine_similarity(v1, v2)[0][0]
    
    return {
        "detected_language": lang,
        "text1": data.text1,
        "text2": data.text2,
        "similarity": round(float(similarity), 4),
        "percentage": f"{round(float(similarity) * 100, 2)}%"
    }
    
@app.post("/embeddings")
async def get_embeddings(data: TextRequest):
    global vector_models, lang_model
    # 1. Определяем, какой это язык
    labels, _ = lang_model.predict(data.text, k=1)
    lang = labels[0].replace("__label__", "")
    
    # 2. Берем модель для этого языка (если нет — берем русскую по дефолту)
    model = vector_models.get(lang, vector_models.get("ru"))
    
    if not model:
        raise HTTPException(status_code=503, detail="Requested vector model not loaded")
    
    vector = model.get_sentence_vector(data.text)
    return {
        "language": lang,
        "embedding": vector.tolist()
    }
    
@app.get("/healthcheck", tags=["System"])
async def health_check():
    """
    Проверка работоспособности сервиса
    """
    return {
        "status": "ok",
        "message": "Service is running"
    }

if __name__ == "__main__":
    import uvicorn
    # Параметры порта должны совпадать с вашим .env или Docker-конфигом
    uvicorn.run(app, host="0.0.0.0", port=8000)