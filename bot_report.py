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
    StopOrderRequest,
)
from alpaca.trading.enums import OrderSide, TimeInForce, OrderStatus

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame


TZ_ET = ZoneInfo("America/New_York")

ALPACA_KEY = os.environ["ALPACA_KEY"]
ALPACA_SECRET = os.environ["ALPACA_SECRET"]
ALPACA_BASE_URL = os.environ.get("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

TG_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
TG_CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]

INVEST_FRACTION = float(os.environ.get("INVEST_FRACTION", "0.80"))
STOP_PCT = float(os.environ.get("STOP_PCT", "0.04"))
MIN_PRICE = float(os.environ.get("MIN_PRICE", "5"))

LOOKBACK_DAYS = int(os.environ.get("LOOKBACK_DAYS", "120"))
BREAKOUT_LOOKBACK = int(os.environ.get("BREAKOUT_LOOKBACK", "20"))
RSI_PERIOD = int(os.environ.get("RSI_PERIOD", "14"))
RSI_MIN = float(os.environ.get("RSI_MIN", "55"))

MARKET_FILTER_SYMBOL = os.environ.get("MARKET_FILTER_SYMBOL", "QQQ")
MARKET_FILTER_MA = int(os.environ.get("MARKET_FILTER_MA", "20"))

SIMULATED_CAPITAL = os.environ.get("SIMULATED_CAPITAL")

TRAIL_ARM_PCT = float(os.environ.get("TRAIL_ARM_PCT", "0.08"))
TRAIL_PCT = float(os.environ.get("TRAIL_PCT", "0.05"))
LOCK_PROFIT_PCT = float(os.environ.get("LOCK_PROFIT_PCT", "0.02"))

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
    if MARKET_FILTER_SYMBOL not in tickers:
        tickers.append(MARKET_FILTER_SYMBOL)
    return tickers


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta.clip(upper=0))
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, pd.NA)
    return 100 - (100 / (1 + rs))


def safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default


def fmt_money(x: float) -> str:
    sign = "+" if x >= 0 else ""
    return f"{sign}${x:,.2f}"


def today_range_et():
    now = dt.datetime.now(TZ_ET)
    start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    end = start + dt.timedelta(days=1)
    return start, end, now


def get_daily_bars(symbols: list[str], start: dt.datetime, end: dt.datetime) -> pd.DataFrame:
    req = StockBarsRequest(
        symbol_or_symbols=symbols,
        timeframe=TimeFrame.Day,
        start=start,
        end=end,
        adjustment="all",
        feed="iex",
    )
    bars = data.get_stock_bars(req).df
    if bars.empty:
        return bars
    bars = bars.reset_index()
    bars["timestamp"] = pd.to_datetime(bars["timestamp"], utc=True).dt.tz_convert(TZ_ET)
    return bars


def round2(x: float) -> float:
    return float(f"{x:.2f}")


def list_open_orders():
    req = GetOrdersRequest(status=OrderStatus.OPEN, limit=200)
    return trading.get_orders(req)


def cancel_open_stop_orders_for_symbol(symbol: str):
    orders = list_open_orders()
    for o in orders:
        try:
            if o.symbol == symbol and str(o.side).lower().endswith("sell") and str(o.order_type).lower().endswith("stop"):
                trading.cancel_order_by_id(o.id)
        except Exception:
            pass


def get_existing_stop_price(symbol: str) -> float | None:
    orders = list_open_orders()
    for o in orders:
        try:
            if o.symbol == symbol and str(o.side).lower().endswith("sell") and str(o.order_type).lower().endswith("stop"):
                sp = getattr(o, "stop_price", None)
                if sp is not None:
                    return safe_float(sp, None)
        except Exception:
            continue
    return None


def place_stop(symbol: str, qty: float, stop_price: float):
    stop = StopOrderRequest(
        symbol=symbol,
        qty=qty,
        side=OrderSide.SELL,
        time_in_force=TimeInForce.GTC,
        stop_price=round2(stop_price),
    )
    trading.submit_order(stop)


def main():
    start_day, end_day, now_et = today_range_et()

    acct = trading.get_account()
    if SIMULATED_CAPITAL:
        equity = float(SIMULATED_CAPITAL)
        cash = float(SIMULATED_CAPITAL)
        last_equity = equity
    else:
        equity = safe_float(acct.equity)
        cash = safe_float(acct.cash)
        last_equity = safe_float(getattr(acct, "last_equity", equity))

    day_pnl = equity - last_equity
    day_pnl_pct = (day_pnl / last_equity * 100.0) if last_equity else 0.0

    universe = load_universe()
    start = (now_et - dt.timedelta(days=LOOKBACK_DAYS + 10)).replace(hour=0, minute=0, second=0, microsecond=0)

    try:
        bars = get_daily_bars(universe, start, now_et)
    except Exception as e:
        telegram_send(f"⚠️ Quantito: error bajando datos: {type(e).__name__}: {str(e)[:180]}")
        return

    def build_indicators(sym: str) -> dict | None:
        df = bars[bars["symbol"] == sym].sort_values("timestamp")
        need = max(MARKET_FILTER_MA, BREAKOUT_LOOKBACK, RSI_PERIOD, 50) + 5
        if df.empty or len(df) < need:
            return None

        close = df["close"].astype(float).reset_index(drop=True)
        ma20 = close.rolling(20).mean()
        ma50 = close.rolling(50).mean()
        rsi_s = rsi(close, RSI_PERIOD)

        last_close = float(close.iloc[-1])
        last_ma20 = float(ma20.iloc[-1])
        last_ma50 = float(ma50.iloc[-1])
        last_rsi = float(rsi_s.iloc[-1])

        prev = close.iloc[:-1]
        if len(prev) < BREAKOUT_LOOKBACK:
            return None
        max_prev_n = float(prev.tail(BREAKOUT_LOOKBACK).max())

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

    mkt = build_indicators(MARKET_FILTER_SYMBOL)
    if not mkt:
        telegram_send("⚠️ Quantito: no pude calcular filtro de mercado (datos insuficientes).")
        return

    if MARKET_FILTER_MA <= 20:
        mkt_ma = mkt["ma20"]
        ma_label = "MA20"
    else:
        mkt_ma = mkt["ma50"]
        ma_label = "MA50"

    market_ok = (mkt["close"] > mkt_ma)

    positions = trading.get_all_positions()
    held_symbol = positions[0].symbol if positions else None

    candidates = []
    for sym in universe:
        if sym == MARKET_FILTER_SYMBOL:
            continue
        ind = build_indicators(sym)
        if not ind:
            continue
        if ind["close"] < MIN_PRICE:
            continue
        signal = (ind["close"] > ind["max_prev_n"] and ind["close"] > ind["ma20"] and ind["rsi"] >= RSI_MIN)
        if signal:
            candidates.append(ind)

    candidates.sort(key=lambda x: (x["ret20_pct"] if not math.isnan(x["ret20_pct"]) else -9999), reverse=True)

    top_lines = [
        f"- {i['symbol']}: close {i['close']:.2f} | MA20 {i['ma20']:.2f} | RSI {i['rsi']:.1f} | ret20 {i['ret20_pct']:.1f}%"
        for i in candidates[:3]
    ] or ["- (sin señales hoy)"]

    action_lines = []
    trailing_lines = []

    if held_symbol:
        pos = positions[0]
        qty = safe_float(pos.qty)
        avg_entry = safe_float(pos.avg_entry_price)
        held_ind = build_indicators(held_symbol)

        if held_ind:
            close = float(held_ind["close"])
            pnl_pct = (close / avg_entry - 1.0) if avg_entry else 0.0

            if (not market_ok) or (close < held_ind["ma20"]):
                reason = "mercado OFF" if not market_ok else "close < MA20"
                try:
                    cancel_open_stop_orders_for_symbol(held_symbol)
                except Exception:
                    pass

                if qty > 0:
                    sell = MarketOrderRequest(symbol=held_symbol, qty=qty, side=OrderSide.SELL, time_in_force=TimeInForce.DAY)
                    trading.submit_order(sell)
                    action_lines.append(f"✅ SELL {held_symbol} qty {qty:g} (salida: {reason})")
                else:
                    action_lines.append(f"ℹ️ Tenía {held_symbol} pero qty inválida, no vendí.")
            else:
                action_lines.append(f"ℹ️ Mantengo {held_symbol} (pnl≈{pnl_pct*100:.1f}%, market_ok={market_ok}).")

                if pnl_pct >= TRAIL_ARM_PCT and qty > 0:
                    existing_stop = get_existing_stop_price(held_symbol)
                    trail_stop = close * (1 - TRAIL_PCT)
                    lock_stop = avg_entry * (1 + LOCK_PROFIT_PCT)
                    new_stop = max(trail_stop, lock_stop, existing_stop or 0.0)

                    if (existing_stop is None) or (new_stop > existing_stop + 0.01):
                        cancel_open_stop_orders_for_symbol(held_symbol)
                        place_stop(held_symbol, qty, new_stop)
                        trailing_lines.append(
                            f"Trailing ON: pnl≈{pnl_pct*100:.1f}% | stop→ {new_stop:.2f} (trail {TRAIL_PCT*100:.1f}% / lock {LOCK_PROFIT_PCT*100:.1f}%)"
                        )
                    else:
                        trailing_lines.append(f"Trailing ON: stop actual {existing_stop:.2f} (pnl≈{pnl_pct*100:.1f}%)")
        else:
            action_lines.append(f"ℹ️ Mantengo {held_symbol} (sin indicadores hoy).")

    if not held_symbol:
        if not market_ok:
            action_lines.append(f"⛔ No compro: filtro mercado OFF ({MARKET_FILTER_SYMBOL} < {ma_label}).")
        else:
            if candidates:
                pick = candidates[0]
                symbol = pick["symbol"]
                notional = max(0.0, cash * INVEST_FRACTION)

                if notional < 5:
                    action_lines.append("⛔ Cash muy bajo para comprar.")
                else:
                    entry_ref = float(pick["close"])
                    initial_stop = round2(entry_ref * (1 - STOP_PCT))

                    buy = MarketOrderRequest(
                        symbol=symbol,
                        notional=round2(notional),
                        side=OrderSide.BUY,
                        time_in_force=TimeInForce.DAY,
                        order_class="oto",
                        stop_loss={"stop_price": initial_stop},
                    )
                    trading.submit_order(buy)
                    action_lines.append(
                        f"✅ BUY {symbol} notional {fmt_money(notional)} | SL inicial {initial_stop} | trailing en +{TRAIL_ARM_PCT*100:.0f}%"
                    )
            else:
                action_lines.append("ℹ️ Mercado OK, pero no hay señales de compra hoy.")

    try:
        req = GetOrdersRequest(status=OrderStatus.ALL, after=start_day, until=end_day, limit=50)
        orders = trading.get_orders(req)
        order_lines = []
        for o in orders[:10]:
            amt = o.qty or o.notional
            order_lines.append(f"- {o.side} {o.symbol} {amt} | {o.status}")
        if not order_lines:
            order_lines = ["- (sin órdenes hoy)"]
    except Exception:
        order_lines = ["- (no pude listar órdenes hoy)"]

    stamp = f"{now_et:%Y-%m-%d %H:%M} ET"
    report = []
    report.append(f"📌 Quantito Daily ({stamp})")
    report.append("")
    report.append("🧭 Filtro mercado")
    report.append(f"- {MARKET_FILTER_SYMBOL}: close {mkt['close']:.2f} | {ma_label} {mkt_ma:.2f}")
    report.append(f"- market_ok: {market_ok}")
    report.append("")
    report.append("💼 Cuenta")
    report.append(f"- Equity: ${equity:,.2f}" + (" (sim)" if SIMULATED_CAPITAL else ""))
    report.append(f"- Cash: ${cash:,.2f}" + (" (sim)" if SIMULATED_CAPITAL else ""))
    report.append(f"- PnL día: {fmt_money(day_pnl)} ({day_pnl_pct:+.2f}%)")
    report.append("")
    report.append("📈 Señales (top)")
    report.extend(top_lines)
    report.append("")
    report.append("⚙️ Acción")
    report.extend([f"- {x}" for x in action_lines] if action_lines else ["- (sin acción)"])

    if trailing_lines:
        report.append("")
        report.append("🟣 Trailing")
        report.extend([f"- {x}" for x in trailing_lines])

    report.append("")
    report.append("🧾 Órdenes hoy")
    report.extend(order_lines)

    telegram_send("\n".join(report))


if __name__ == "__main__":
    main()
