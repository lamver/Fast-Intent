import fasttext
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Создаем экземпляр приложения
app = FastAPI(title="NLP Service")

# Глобальные переменные для моделей
vector_model = None
lang_model = None

@app.on_event("startup")
def load_models():
    global vector_model, lang_model
    try:
        # Загружаем обе модели при старте
        #vector_model = fasttext.load_model("cc.ru.300.bin")
        lang_model = fasttext.load_model("lid.176.bin")
        print("All models loaded successfully")
    except Exception as e:
        print(f"Error loading models: {e}")
        
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
    if vector_model is None:
        raise HTTPException(status_code=503, detail="Vector model not loaded")
    
    # 1. Получаем векторы для обеих фраз
    # get_sentence_vector возвращает одномерный массив (300,)
    v1 = vector_model.get_sentence_vector(data.text1).reshape(1, -1)
    v2 = vector_model.get_sentence_vector(data.text2).reshape(1, -1)
    
    # 2. Считаем косинусное сходство (результат от 0 до 1)
    similarity = cosine_similarity(v1, v2)[0][0]
    
    return {
        "text1": data.text1,
        "text2": data.text2,
        "similarity": round(float(similarity), 4),
        "percentage": f"{round(float(similarity) * 100, 2)}%"
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