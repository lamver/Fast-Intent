from fastapi import FastAPI

# Создаем экземпляр приложения
# Swagger будет доступен по адресу /docs
app = FastAPI(
    title="TTS Service",
    description="Простой сервис для синтеза речи",
    version="1.0.0"
)

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