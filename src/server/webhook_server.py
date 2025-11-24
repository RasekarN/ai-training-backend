import os
import time
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.data_loader.yahoo_loader import YahooDataLoader
from src.models.train_model import ModelTrainer

app = FastAPI()

# Allow all CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cache storage
CACHE = {}
CACHE_TTL = 10  # seconds


class SignalRequest(BaseModel):
    symbol: str | None = None
    ticker: str | None = None
    timeframe: str


YAHOO_MAP = {
    "NIFTY": "^NSEI",
    "BANKNIFTY": "^NSEBANK",
    "CRUDE": "CL=F",
    "NG": "NG=F",
}

TIMEFRAME_MAP = {
    "1D": ("1d", "5y"),
    "1H": ("60m", "730d"),
    "15m": ("15m", "60d"),
    "5m": ("5m", "30d"),
}

loader = YahooDataLoader()


def get_cached_response(key):
    """Return cached response if still valid."""
    if key in CACHE:
        ts, data = CACHE[key]
        if time.time() - ts <= CACHE_TTL:
            print(f"[CACHE HIT] {key}")
            return data
    return None


def set_cache(key, data):
    CACHE[key] = (time.time(), data)


@app.post("/tv-webhook")
async def get_signal(req: SignalRequest):

    # Resolve symbol
    symbol = (req.symbol or req.ticker).upper()
    if symbol not in YAHOO_MAP:
        return {"error": f"Unknown symbol {symbol}"}

    yahoo_symbol = YAHOO_MAP[symbol]

    # Resolve timeframe
    if req.timeframe not in TIMEFRAME_MAP:
        return {"error": f"Invalid timeframe: {req.timeframe}"}

    interval, period = TIMEFRAME_MAP[req.timeframe]

    # Cache Key
    cache_key = f"{symbol}_{interval}"

    # Try Cache First
    cached = get_cached_response(cache_key)
    if cached:
        return cached

    # ============================
    #     EXPONENTIAL BACKOFF
    # ============================
    df = None
    for attempt in range(4):
        try:
            print(f"[Yahoo Fetch Attempt {attempt+1}/4] {yahoo_symbol}")
            df = loader.fetch(yahoo_symbol, interval=interval, period=period)
            if df is not None and len(df) > 100:
                break
        except Exception as e:
            print(f"Yahoo error: {e}")

        # Exponential delay
        time.sleep(0.5 * (2 ** attempt))

    if df is None or len(df) < 100:
        # If Yahoo failed, fallback to cached data
        cached = get_cached_response(cache_key)
        if cached:
            print("[FALLBACK] Using cached data due to Yahoo failure.")
            return cached
        return {"error": f"Yahoo failed for {symbol}, no cache available"}

    # Train & Predict
    trainer = ModelTrainer()
    model = trainer.train(df)
    signal, p_up, p_down, latest_price = trainer.predict_latest_from_df(model, df)

    # Final Response
    response = {
        "input_symbol": symbol,
        "yahoo_symbol": yahoo_symbol,
        "timeframe": req.timeframe,
        "ai_signal": signal,
        "ai_prob_up": p_up,
        "ai_prob_down": p_down,
        "latest_price": latest_price,
        "ohlc": df.tail(200).to_dict(orient="records"),
    }

    set_cache(cache_key, response)
    return response


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.server.webhook_server:app", host="0.0.0.0", port=8000, reload=True)
