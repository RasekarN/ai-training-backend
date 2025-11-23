import os
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

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
#  Payload (now accepts both "symbol" and "ticker")
# ============================================
class SignalRequest(BaseModel):
    symbol: str | None = None       # new param
    ticker: str | None = None       # old param (TradingView style)
    timeframe: str                  # required


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


# ============================================
#            Main Prediction Route
# ============================================
@app.post("/tv-webhook")
async def get_signal(req: SignalRequest):

    print("\n========== NEW SIGNAL REQUEST ==========")
    print(f"Raw Payload: {req}")

    # --------------------------------------------
    # 1) Resolve symbol (NEW or OLD input)
    # --------------------------------------------
    symbol = req.symbol or req.ticker
    if symbol is None:
        return {"error": "Missing field: symbol or ticker is required"}

    symbol = symbol.upper()

    print(f"Resolved Symbol = {symbol}")

    # --------------------------------------------
    # 2) Validate symbol
    # --------------------------------------------
    if symbol not in YAHOO_MAP:
        return {"error": f"Unknown symbol: {symbol}. Allowed: {list(YAHOO_MAP.keys())}"}

    yahoo_symbol = YAHOO_MAP[symbol]

    # --------------------------------------------
    # 3) Validate timeframe
    # --------------------------------------------
    if req.timeframe not in TIMEFRAME_MAP:
        return {"error": f"Invalid timeframe: {req.timeframe}"}

    interval, period = TIMEFRAME_MAP[req.timeframe]

    print(f"Yahoo Symbol → {yahoo_symbol}")
    print(f"Interval → {interval}, Period → {period}")

    # --------------------------------------------
    # 4) Fetch historical data
    # --------------------------------------------
    try:
        df = loader.fetch(yahoo_symbol, interval=interval, period=period)
    except Exception as e:
        return {"error": f"Yahoo fetch failed: {str(e)}"}

    if df is None or len(df) < 120:
        return {"error": f"Insufficient data returned for {symbol}"}

    print(f"Data Loaded: {len(df)} rows")

    # --------------------------------------------
    # 5) Train model for this asset + timeframe
    # --------------------------------------------
    trainer = ModelTrainer()

    try:
        model = trainer.train(df)
    except Exception as e:
        print(f"Training Error: {str(e)}")
        return {"error": f"Training failed: {str(e)}"}

    print("Training COMPLETE.")

    # --------------------------------------------
    # 6) Predict next move
    # --------------------------------------------
    try:
        signal, p_up, p_down, latest_price = trainer.predict_latest_from_df(model, df)
    except Exception as e:
        print(f"Prediction Error: {str(e)}")
        return {"error": f"Prediction failed: {str(e)}"}

    print(f"Signal = {signal} | UP={p_up:.4f} DOWN={p_down:.4f}")
    print("==========================================\n")

    # --------------------------------------------
    # 7) FINAL RESPONSE
    # --------------------------------------------
    return {
        "input_symbol": symbol,
        "yahoo_symbol": yahoo_symbol,
        "timeframe": req.timeframe,
        "ai_signal": signal,
        "ai_prob_up": p_up,
        "ai_prob_down": p_down,
        "latest_price": latest_price,
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