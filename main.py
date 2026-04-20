import os
import fasttext
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import List, Dict
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
import asyncio
from security import check_ip_middleware, update_remote_ips_task, force_update_ips, dynamic_ips

app = FastAPI(title="NLP Service")
app.middleware("http")(check_ip_middleware)

@app.on_event("startup")
async def startup_ip_task():
    # Запускаем фоновое обновление IP
    asyncio.create_task(update_remote_ips())

# ПУТИ И МОДЕЛИ
BASE_PATH = "/app/models"
vector_models = {}
lang_model = None

# ГЛОБАЛЬНАЯ ЗАГРУЗКА (Для корректной работы Gunicorn --preload)
try:
    print("--- Инициализация моделей ---")
    # 1. Загрузка определителя языка
    lid_path = os.path.join(BASE_PATH, "lid.176.bin")
    if os.path.exists(lid_path):
        lang_model = fasttext.load_model(lid_path)
        print("LID model: OK")

    # 2. Загрузка векторов
    for lang in ["ru", "en", "es"]:
        path = os.path.join(BASE_PATH, f"cc.{lang}.300.bin")
        if os.path.exists(path):
            vector_models[lang] = fasttext.load_model(path)
            print(f"Vector model {lang}: OK")
    print(f"Загружено моделей: {list(vector_models.keys())}")
except Exception as e:
    print(f"ОШИБКА ЗАГРУЗКИ: {e}")

class TextRequest(BaseModel):
    text: str

class CompareRequest(BaseModel):
    text1: str
    text2: str

def get_text_lang(text: str):
    """Вспомогательная функция для получения чистого кода языка"""
    if lang_model is None:
        return "ru"
    # Fasttext возвращает (('__label__ru',), array([0.99]))
    prediction = lang_model.predict(text, k=1)
    return prediction[0][0].replace("__label__", "")

# Загружаем настройки из окружения
ENV_ALLOWED_IPS = os.getenv("ALLOWED_IPS", "127.0.0.1").split(",")
REMOTE_IPS_URL = os.getenv("REMOTE_IPS_URL")

# Глобальный кэш для динамических IP
dynamic_ips = set()

@app.post("/detect-language", tags=["NLP"])
async def detect_language(data: TextRequest):
    if lang_model is None:
        raise HTTPException(status_code=503, detail="LID not loaded")
    
    labels, probabilities = lang_model.predict(data.text, k=3)
    results = []
    for l, p in zip(labels, probabilities):
        results.append({
            "language": l.replace("__label__", ""),
            "confidence": round(float(p), 4)
        })
    return {"text": data.text, "predictions": results}

@app.post("/compare-vectors", tags=["NLP"])
async def compare_texts(data: CompareRequest):
    # Определяем язык
    lang = get_text_lang(data.text1)
    
    # ВАЖНО: берем модель из словаря. Если нет - пробуем RU, если нет - берем любую доступную
    model = vector_models.get(lang) or vector_models.get("ru")
    if not model and vector_models:
        model = list(vector_models.values())[0]

    if not model:
        raise HTTPException(status_code=503, detail=f"Models dict is empty. Seen files: {os.listdir(BASE_PATH)}")
    
    v1 = model.get_sentence_vector(data.text1).reshape(1, -1)
    v2 = model.get_sentence_vector(data.text2).reshape(1, -1)
    similarity = cosine_similarity(v1, v2)[0][0]
    
    return {
        "detected_language": lang,
        "similarity": round(float(similarity), 4),
        "percentage": f"{round(float(similarity) * 100, 2)}%"
    }

class RouterRequest(BaseModel):
    text: str
    intents: Dict[str, List[str]]  # Словарь: {"movies": ["ужасы", "кино"], "music": ["треки"]}

@app.post("/route-intent", tags=["NLP"])
async def route_intent(data: RouterRequest):
    if not lang_model or not vector_models:
        raise HTTPException(status_code=503, detail="Модели не загружены")

    # 1. Определяем язык и выбираем модель
    lang = get_text_lang(data.text)
    model = vector_models.get(lang) or vector_models.get("ru")
    
    # 2. Вектор входного текста пользователя
    user_vec = model.get_sentence_vector(data.text).reshape(1, -1)
    
    all_predictions = []

    # 3. Проходим по всем переданным интентам и считаем сходство
    for intent_name, examples in data.intents.items():
        if not examples:
            continue
            
        # Считаем векторы для всех примеров-фраз в этом интенте
        example_vecs = np.array([model.get_sentence_vector(ex) for ex in examples])
        
        # Находим косинусное сходство между текстом юзера и всеми примерами интента
        scores = cosine_similarity(user_vec, example_vecs)
        
        # Берем максимальное значение (насколько близко подошел лучший пример из группы)
        max_intent_score = float(np.max(scores))
        
        all_predictions.append({
            "intent": intent_name,
            "confidence": round(max_intent_score, 4)
        })

    # 4. Сортируем результаты по убыванию уверенности
    all_predictions.sort(key=lambda x: x["confidence"], reverse=True)

    # 5. Если ничего не нашли (список пуст)
    if not all_predictions:
        return {"text": data.text, "intent": "fallback", "message": "Список интентов пуст"}

    best = all_predictions[0]
    
    # Считаем "разрыв" до второго места (помогает понять, насколько бот уверен в выборе)
    gap = 0
    if len(all_predictions) > 1:
        gap = round(best["confidence"] - all_predictions[1]["confidence"], 4)

    # 6. Порог отсечения (Threshold)
    threshold = 0.4
    if best["confidence"] < threshold:
        return {
            "text": data.text,
            "intent": "fallback",
            "confidence": best["confidence"],
            "all_predictions": all_predictions,
            "message": "Ни один интент не преодолел порог уверенности"
        }

    return {
        "text": data.text,
        "intent": best["intent"],
        "confidence": best["confidence"],
        "gap_to_second": gap,
        "all_predictions": all_predictions
    }
    
@app.get("/debug-models", tags=["System"])
async def debug_models():
    import os
    base_path = "/app/models"
    debug_info = []
    
    if os.path.exists(base_path):
        for filename in os.listdir(base_path):
            file_path = os.path.join(base_path, filename)
            
            # Получаем метаданные файла
            stat = os.stat(file_path)
            size_mb = round(stat.st_size / (1024 * 1024), 2)
            # Превращаем время в нормальную дату
            mtime = datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
            
            debug_info.append({
                "file": filename,
                "size_mb": size_mb,
                "modified": mtime
            })
    else:
        debug_info = "Folder /app/models not found"

    return {
        "files_info": debug_info,
        "loaded_keys_in_dict": list(vector_models.keys()),
        "lang_model_loaded": lang_model is not None,
        "current_working_dir": os.getcwd()
    }

@app.get("/healthcheck", tags=["System"])
async def health_check():
    return {"status": "ok", "models_count": len(vector_models)}

@app.get("/refresh-ips", tags=["System"])
async def refresh_ips_endpoint(token: str = None):
    # Добавим простую защиту, чтобы кто попало не дергал обновление
    # Сверь токен из URL с тем, что в .env (например, REFRESH_TOKEN=super-secret)
    if token != os.getenv("REFRESH_TOKEN"):
        raise HTTPException(status_code=401, detail="Invalid token")

    try:
        # Вызываем нашу функцию загрузки (без бесконечного цикла, просто один раз)
        success = await force_update_ips()
        if success:
            return {"status": "success", "updated_ips": list(dynamic_ips)}
        else:
            return {"status": "error", "message": "Check server logs"}
    except Exception as e:
        return {"status": "error", "detail": str(e)}