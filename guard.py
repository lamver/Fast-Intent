import os
import httpx
import asyncio
from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse

# Настройки из ENV
ENV_ALLOWED_IPS = os.getenv("ALLOWED_IPS", "127.0.0.1,192.168.32.1").split(",")
REMOTE_IPS_URL = os.getenv("REMOTE_IPS_URL")
REFRESH_TOKEN = os.getenv("REFRESH_TOKEN", "default-secret")

# Кэш в памяти
dynamic_ips = set()

async def force_update_ips():
    """Одиночное обновление списка IP из внешнего источника"""
    global dynamic_ips
    if not REMOTE_IPS_URL:
        return False
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(REMOTE_IPS_URL, timeout=10.0)
            if response.status_code == 200:
                new_ips = response.text.replace("\n", ",").split(",")
                dynamic_ips = {ip.strip() for ip in new_ips if ip.strip()}
                return True
    except Exception as e:
        print(f"Error fetching remote IPs: {e}")
    return False

async def update_remote_ips_task():
    """Фоновая задача (бесконечный цикл)"""
    while True:
        await force_update_ips()
        await asyncio.sleep(600)  # 10 минут

async def check_ip_middleware(request: Request, call_next):
    """Логика проверки IP для Middleware"""
    # Белый список путей (без проверки)
    whitelist_paths = ["/healthcheck", "/refresh-ips", "/debug-models", "/docs", "/openapi.json"]
    
    if request.url.path in whitelist_paths:
        return await call_next(request)

    # Определяем IP
    x_forwarded_for = request.headers.get("x-forwarded-for")
    client_ip = x_forwarded_for.split(",")[0].strip() if x_forwarded_for else request.client.host

    # Проверка
    if client_ip not in ENV_ALLOWED_IPS and client_ip not in dynamic_ips:
        return JSONResponse(
            status_code=403,
            content={"detail": f"Access denied: IP {client_ip} not authorized"}
        )
    
    return await call_next(request)
