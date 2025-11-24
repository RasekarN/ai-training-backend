import os
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import time

from src.data_loader.yahoo_loader import YahooDataLoader
from src.models.train_model import ModelTrainer

app = FastAPI()

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================
#  Payload
# ============================================
class SignalRequest(BaseModel):
    symbol: str | None = None
    ticker: str | None = None
    timeframe: str


# MAP INDEX → YAHOO SYMBOLS
YAHOO_MAP = {
    "NIFTY": "^NSEI",
    "BANKNIFTY": "^NSEBANK",
    "CRUDE": "CL=F",
    "NG": "NG=F",
}

# MAP FRONTEND TIMEFRAME → YAHOO INTERVAL + PERIOD
TIMEFRAME_MAP = {
    "1D": ("1d", "5y"),
    "1H": ("60m", "730d"),
    "15m": ("15m", "60d"),
    "5m": ("5m", "30d"),
}

loader = YahooDataLoader()


# ======================================================================
#         **RENDER-FIX**: RESILIENT DATA FETCHER (NO EMPTY DF EVER)
# ======================================================================
def safe_fetch_yahoo(symbol, interval, period, attempts=4):
    """
    Render often blocks Yahoo requests. Retry several times.
    """

    for i in range(attempts):
        try:
            print(f"[Yahoo Fetch Attempt {i+1}/{attempts}] {symbol} interval={interval}")
            df = loader.fetch(symbol, interval=interval, period=period)

            if df is not None and len(df) > 50:
                return df

        except Exception as e:
            print(f"Fetch error: {e}")

        time.sleep(1.2)   # Avoid Yahoo rate limits

    print("⚠ Yahoo failed — final attempt using fallback interval...")

    # FINAL FALLBACK (Guaranteed)
    try:
        fallback_df = loader.fetch(symbol, interval="1d", period="1y")
        if fallback_df is not None and len(fallback_df) > 50:
            return fallback_df
    except:
        pass

    return None


# ============================================
#            Main Prediction Route
# ============================================
@app.post("/tv-webhook")
async def get_signal(req: SignalRequest):

    print("\n========== NEW SIGNAL REQUEST ==========")
    print(f"Raw Payload: {req}")

    # --------------------------------------------
    # 1) Resolve symbol
    # --------------------------------------------
    symbol = req.symbol or req.ticker
    if symbol is None:
        return {"error": "Missing field: symbol or ticker is required"}

    symbol = symbol.upper()
    print(f"Resolved Symbol = {symbol}")

    if symbol not in YAHOO_MAP:
        return {"error": f"Unknown symbol {symbol}. Allowed: {list(YAHOO_MAP.keys())}"}

    yahoo_symbol = YAHOO_MAP[symbol]

    # --------------------------------------------
    # 2) Validate timeframe
    # --------------------------------------------
    if req.timeframe not in TIMEFRAME_MAP:
        return {"error": f"Invalid timeframe {req.timeframe}"}

    interval, period = TIMEFRAME_MAP[req.timeframe]

    print(f"Yahoo Symbol → {yahoo_symbol}")
    print(f"Interval → {interval}, Period → {period}")

    # --------------------------------------------
    # 3) Fetch data (Render-safe)
    # --------------------------------------------
    df = safe_fetch_yahoo(yahoo_symbol, interval, period)

    if df is None or len(df) < 50:
        print("❌ FATAL: No data even after retries")
        return {"error": f"Yahoo returned insufficient data for {symbol}. Try later."}

    print(f"Data Loaded: {len(df)} rows")

    # Clean any NaN for safety
    df = df.dropna().reset_index(drop=True)

    # --------------------------------------------
    # 4) Train model
    # --------------------------------------------
    trainer = ModelTrainer()

    try:
        model = trainer.train(df)
    except Exception as e:
        print(f"Training Error: {e}")
        return {"error": f"Training failed: {str(e)}"}

    print("Training COMPLETE.")

    # --------------------------------------------
    # 5) Predict
    # --------------------------------------------
    try:
        signal, p_up, p_down, latest_price = trainer.predict_latest_from_df(model, df)
    except Exception as e:
        print(f"Prediction Error: {e}")
        return {"error": f"Prediction failed: {str(e)}"}

    print(f"Signal = {signal} | UP={p_up:.4f} DOWN={p_down:.4f}")
    print("==========================================\n")

    return {
        "input_symbol": symbol,
        "yahoo_symbol": yahoo_symbol,
        "timeframe": req.timeframe,
        "ai_signal": signal,
        "ai_prob_up": float(p_up),
        "ai_prob_down": float(p_down),
        "latest_price": float(latest_price),
        "ohlc": df.tail(200).to_dict(orient="records")
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.server.webhook_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
