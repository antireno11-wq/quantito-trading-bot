import os
import math
import datetime as dt
from zoneinfo import ZoneInfo

import pandas as pd
import requests

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (
    MarketOrderRequest,
    GetOrdersRequest,
)
from alpaca.trading.enums import (
    OrderSide,
    TimeInForce,
    OrderStatus,
)
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame


TZ_ET = ZoneInfo("America/New_York")

# ===== ENV =====
ALPACA_KEY = os.environ["ALPACA_KEY"]
ALPACA_SECRET = os.environ["ALPACA_SECRET"]
ALPACA_BASE_URL = os.environ.get("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

TG_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
TG_CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]

# ===== SETTINGS (ajustables por variables de entorno) =====
INVEST_FRACTION = float(os.environ.get("INVEST_FRACTION", "0.80"))  # 80% del cash
STOP_PCT = float(os.environ.get("STOP_PCT", "0.04"))               # -4%
TAKE_PCT = float(os.environ.get("TAKE_PCT", "0.08"))               # +8%
MIN_PRICE = float(os.environ.get("MIN_PRICE", "5"))                # evitar cosas muy baratas
LOOKBACK_DAYS = int(os.environ.get("LOOKBACK_DAYS", "120"))         # para MA50 y señales
BREAKOUT_LOOKBACK = int(os.environ.get("BREAKOUT_LOOKBACK", "20"))  # máximo 20 días
RSI_PERIOD = int(os.environ.get("RSI_PERIOD", "14"))
RSI_MIN = float(os.environ.get("RSI_MIN", "55"))

MARKET_FILTER_SYMBOL = os.environ.get("MARKET_FILTER_SYMBOL", "QQQ")  # filtro
MARKET_FILTER_MA = int(os.environ.get("MARKET_FILTER_MA", "50"))

PAPER = ("paper" in ALPACA_BASE_URL)

trading = TradingClient(ALPACA_KEY, ALPACA_SECRET, paper=PAPER)
data = StockHistoricalDataClient(ALPACA_KEY, ALPACA_SECRET)


def telegram_send(text: str) -> None:
    url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
    r = requests.post(url, json={"chat_id": TG_CHAT_ID, "text": text}, timeout=20)
    r.raise_for_status()


def load_universe() -> list[str]:
    with open("universe.txt", "r", encoding="utf-8") as f:
        tickers = [x.strip().upper() for x in f.readlines()]
    tickers = [t for t in tickers if t and not t.startswith("#")]
    # Aseguramos que el filtro esté incluido aunque lo borres del archivo
    if MARKET_FILTER_SYMBOL not in tickers:
        tickers.append(MARKET_FILTER_SYMBOL)
    return tickers


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, pd.NA)
    return 100 - (100 / (1 + rs))


def get_daily_bars(symbols: list[str], start: dt.datetime, end: dt.datetime) -> pd.DataFrame:
    req = StockBarsRequest(
        symbol_or_symbols=symbols,
        timeframe=TimeFrame.Day,
        start=start,
        end=end,
        adjustment="all",
    )
    bars = data.get_stock_bars(req).df
    # df index: MultiIndex (symbol, timestamp)
    if bars.empty:
        return bars
    bars = bars.reset_index()
    bars["timestamp"] = pd.to_datetime(bars["timestamp"], utc=True).dt.tz_convert(TZ_ET)
    return bars


def today_range_et():
    now = dt.datetime.now(TZ_ET)
    start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    end = start + dt.timedelta(days=1)
    return start, end, now


def safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default


def fmt_money(x: float) -> str:
    sign = "+" if x >= 0 else ""
    return f"{sign}${x:,.2f}"


def main():
    start_day, end_day, now_et = today_range_et()

    # ---- Account ----
    acct = trading.get_account()
    equity = safe_float(acct.equity)
    cash = safe_float(acct.cash)
    last_equity = safe_float(getattr(acct, "last_equity", equity))
    day_pnl = equity - last_equity
    day_pnl_pct = (day_pnl / last_equity * 100.0) if last_equity else 0.0

    # ---- Universe + data ----
    universe = load_universe()
    start = (now_et - dt.timedelta(days=LOOKBACK_DAYS + 10)).replace(hour=0, minute=0, second=0, microsecond=0)
    end = now_et

    try:
        bars = get_daily_bars(universe, start, end)
    except Exception as e:
        telegram_send(f"⚠️ Quantito: error bajando datos: {type(e).__name__}: {str(e)[:180]}")
        raise

    # Helper: build indicator table per symbol
    def build_indicators(sym: str) -> dict | None:
        df = bars[bars["symbol"] == sym].sort_values("timestamp")
        if df.empty or len(df) < max(MARKET_FILTER_MA, BREAKOUT_LOOKBACK, RSI_PERIOD) + 5:
            return None

        close = df["close"].astype(float).reset_index(drop=True)
        ma20 = close.rolling(20).mean()
        ma50 = close.rolling(50).mean()
        rsi14 = rsi(close, RSI_PERIOD)

        last_close = float(close.iloc[-1])
        last_ma20 = float(ma20.iloc[-1])
        last_ma50 = float(ma50.iloc[-1])
        last_rsi = float(rsi14.iloc[-1])

        # breakout: close > max prev N closes (excluding today)
        prev = close.iloc[:-1]
        if len(prev) < BREAKOUT_LOOKBACK:
            return None
        max_prev_n = float(prev.tail(BREAKOUT_LOOKBACK).max())

        # simple momentum score: 20d return
        ret20 = float((close.iloc[-1] / close.iloc[-21] - 1.0) * 100.0) if len(close) > 21 else float("nan")

        return {
            "symbol": sym,
            "close": last_close,
            "ma20": last_ma20,
            "ma50": last_ma50,
            "rsi": last_rsi,
            "max_prev_n": max_prev_n,
            "ret20_pct": ret20,
        }

    # ---- Market filter (QQQ > MA50) ----
    mkt = build_indicators(MARKET_FILTER_SYMBOL)
    if not mkt:
        telegram_send("⚠️ Quantito: no pude calcular filtro de mercado (datos insuficientes).")
        return

    market_ok = (mkt["close"] > mkt["ma50"])

    # ---- Current positions ----
    positions = trading.get_all_positions()
    held_symbol = positions[0].symbol if positions else None

    # ---- Decide exits/entries ----
    action_lines = []
    top_lines = []

    # candidate scan
    candidates = []
    for sym in universe:
        if sym == MARKET_FILTER_SYMBOL:
            continue
        ind = build_indicators(sym)
        if not ind:
            continue
        if ind["close"] < MIN_PRICE:
            continue

        signal = (
            ind["close"] > ind["max_prev_n"] and
            ind["close"] > ind["ma20"] and
            ind["rsi"] >= RSI_MIN
        )
        if signal:
            candidates.append(ind)

    # sort by 20d return (best first)
    candidates.sort(key=lambda x: (x["ret20_pct"] if not math.isnan(x["ret20_pct"]) else -9999), reverse=True)

    # build top3 display
    for ind in candidates[:3]:
        top_lines.append(
            f"- {ind['symbol']}: close {ind['close']:.2f} | MA20 {ind['ma20']:.2f} | RSI {ind['rsi']:.1f} | ret20 {ind['ret20_pct']:.1f}%"
        )
    if not top_lines:
        top_lines = ["- (sin señales hoy)"]

    # Exit logic (si hay posición)
    if held_symbol:
        held_ind = build_indicators(held_symbol)
        exit_reason = None
        if not held_ind:
            exit_reason = "no pude calcular indicadores, mantengo posición"
        else:
            # salida si pierde MA20 o si filtro mercado se pone malo
            if held_ind["close"] < held_ind["ma20"]:
                exit_reason = "salida: close < MA20"
            elif not market_ok:
                exit_reason = "salida: mercado (QQQ) bajo MA50"

        if exit_reason and "salida:" in exit_reason:
            # submit SELL for next open (market order)
            qty = safe_float(positions[0].qty)
            if qty > 0:
                sell = MarketOrderRequest(
                    symbol=held_symbol,
                    qty=qty,
                    side=OrderSide.SELL,
                    time_in_force=TimeInForce.DAY,
                )
                trading.submit_order(sell)
                action_lines.append(f"✅ SELL {held_symbol} qty {qty:g} ({exit_reason})")
            else:
                action_lines.append(f"ℹ️ Tenía posición {held_symbol} pero qty inválida, no vendí.")
        else:
            action_lines.append(f"ℹ️ Mantengo {held_symbol} (market_ok={market_ok}).")

    # Entry logic (solo si no hay posición)
    if not held_symbol:
        if not market_ok:
            action_lines.append("⛔ No compro: filtro mercado OFF (QQQ < MA50).")
        else:
            if candidates:
                pick = candidates[0]
                symbol = pick["symbol"]

                notional = max(0.0, cash * INVEST_FRACTION)
                if notional < 5:
                    action_lines.append("⛔ Cash muy bajo para comprar.")
                else:
                    entry = float(pick["close"])
                    stop_price = round(entry * (1 - STOP_PCT), 2)
                    take_price = round(entry * (1 + TAKE_PCT), 2)

                    buy = MarketOrderRequest(
                        symbol=symbol,
                        notional=round(notional, 2),
                        side=OrderSide.BUY,
                        time_in_force=TimeInForce.DAY,
                        # bracket: Alpaca crea TP/SL asociados
                        order_class="bracket",
                        take_profit={"limit_price": take_price},
                        stop_loss={"stop_price": stop_price},
                    )
                    trading.submit_order(buy)
                    action_lines.append(
                        f"✅ BUY {symbol} notional {fmt_money(notional)} | TP {take_price} | SL {stop_price}"
                    )
            else:
                action_lines.append("ℹ️ Mercado OK, pero no hay señales de compra hoy.")

    # ---- Orders today ----
    try:
        req = GetOrdersRequest(status=OrderStatus.ALL, after=start_day, until=end_day, limit=50)
        orders = trading.get_orders(req)
        order_lines = []
        for o in orders[:10]:
            order_lines.append(f"- {o.side} {o.symbol} {o.qty or o.notional} | {o.status}")
        if not order_lines:
            order_lines = ["- (sin órdenes hoy)"]
    except Exception:
        order_lines = ["- (no pude listar órdenes hoy)"]

    # ---- Build report ----
    stamp = f"{now_et:%Y-%m-%d %H:%M} ET"
    report = []
    report.append(f"📌 Quantito Daily ({stamp})")
    report.append("")
    report.append("🧭 Filtro mercado")
    report.append(f"- {MARKET_FILTER_SYMBOL}: close {mkt['close']:.2f} | MA{MARKET_FILTER_MA} {mkt['ma50']:.2f}")
    report.append(f"- market_ok: {market_ok}")
    report.append("")
    report.append("💼 Cuenta")
    report.append(f"- Equity: ${equity:,.2f}")
    report.append(f"- Cash: ${cash:,.2f}")
    report.append(f"- PnL día: {fmt_money(day_pnl)} ({day_pnl_pct:+.2f}%)")
    report.append("")
    report.append("📈 Señales (top)")
    report.extend(top_lines)
    report.append("")
    report.append("⚙️ Acción")
    report.extend([f"- {x}" for x in action_lines] if action_lines else ["- (sin acción)"])
    report.append("")
    report.append("🧾 Órdenes hoy")
    report.extend(order_lines)

    telegram_send("\n".join(report))


if __name__ == "__main__":
    main()
