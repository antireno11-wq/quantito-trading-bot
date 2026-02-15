import os
import math
import sqlite3
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

# ===== ENV =====
ALPACA_KEY = os.environ["ALPACA_KEY"]
ALPACA_SECRET = os.environ["ALPACA_SECRET"]
ALPACA_BASE_URL = os.environ.get("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

TG_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
TG_CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]

DB_PATH = os.environ.get("DB_PATH", "quantito.sqlite")

# ===== SETTINGS =====
INVEST_FRACTION = float(os.environ.get("INVEST_FRACTION", "0.80"))
STOP_PCT = float(os.environ.get("STOP_PCT", "0.04"))
MIN_PRICE = float(os.environ.get("MIN_PRICE", "5"))

LOOKBACK_DAYS = int(os.environ.get("LOOKBACK_DAYS", "120"))
BREAKOUT_LOOKBACK = int(os.environ.get("BREAKOUT_LOOKBACK", "20"))
RSI_PERIOD = int(os.environ.get("RSI_PERIOD", "14"))
RSI_MIN = float(os.environ.get("RSI_MIN", "55"))

MARKET_FILTER_SYMBOL = os.environ.get("MARKET_FILTER_SYMBOL", "QQQ")
MARKET_FILTER_MA = int(os.environ.get("MARKET_FILTER_MA", "20"))

SIMULATED_CAPITAL = os.environ.get("SIMULATED_CAPITAL")  # e.g. "200"

# Trailing behavior (daily)
TRAIL_ARM_PCT = float(os.environ.get("TRAIL_ARM_PCT", "0.08"))      # +8% activates trailing
TRAIL_PCT = float(os.environ.get("TRAIL_PCT", "0.05"))              # trailing 5% from close
LOCK_PROFIT_PCT = float(os.environ.get("LOCK_PROFIT_PCT", "0.02"))  # lock at least +2% once armed

PAPER = ("paper" in ALPACA_BASE_URL)

trading = TradingClient(ALPACA_KEY, ALPACA_SECRET, paper=PAPER)
data = StockHistoricalDataClient(ALPACA_KEY, ALPACA_SECRET)


# =========================
# Helpers
# =========================
def telegram_send(text: str) -> None:
    url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
    r = requests.post(url, json={"chat_id": TG_CHAT_ID, "text": text}, timeout=20)
    r.raise_for_status()


def safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default


def round2(x: float) -> float:
    return float(f"{x:.2f}")


def fmt_money(x: float) -> str:
    sign = "+" if x >= 0 else ""
    return f"{sign}${x:,.2f}"


def today_range_et():
    now = dt.datetime.now(TZ_ET)
    start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    end = start + dt.timedelta(days=1)
    return start, end, now


def month_start_et(now_et: dt.datetime) -> dt.datetime:
    return now_et.replace(day=1, hour=0, minute=0, second=0, microsecond=0)


def ensure_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True) if "/" in DB_PATH else None
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS kv (
            k TEXT PRIMARY KEY,
            v TEXT NOT NULL
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS orders_filled (
            order_id TEXT PRIMARY KEY,
            filled_at TEXT,
            symbol TEXT,
            side TEXT,
            qty REAL,
            filled_avg_price REAL
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS trades_closed (
            trade_key TEXT PRIMARY KEY,
            symbol TEXT,
            entry_time TEXT,
            exit_time TEXT,
            qty REAL,
            entry_price REAL,
            exit_price REAL,
            pnl REAL,
            pnl_pct REAL
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS equity_curve (
            day TEXT PRIMARY KEY,   -- YYYY-MM-DD (ET)
            equity REAL
        )
    """)
    con.commit()
    con.close()


def kv_get(key: str, default: str | None = None) -> str | None:
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("SELECT v FROM kv WHERE k=?", (key,))
    row = cur.fetchone()
    con.close()
    return row[0] if row else default


def kv_set(key: str, value: str):
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute(
        "INSERT INTO kv(k,v) VALUES(?,?) ON CONFLICT(k) DO UPDATE SET v=excluded.v",
        (key, value),
    )
    con.commit()
    con.close()


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


def get_daily_bars(symbols: list[str], start: dt.datetime, end: dt.datetime) -> pd.DataFrame:
    req = StockBarsRequest(
        symbol_or_symbols=symbols,
        timeframe=TimeFrame.Day,
        start=start,
        end=end,
        adjustment="all",
        feed="iex",  # avoids SIP restriction
    )
    bars = data.get_stock_bars(req).df
    if bars.empty:
        return bars
    bars = bars.reset_index()
    bars["timestamp"] = pd.to_datetime(bars["timestamp"], utc=True).dt.tz_convert(TZ_ET)
    return bars


def list_open_orders():
    req = GetOrdersRequest(status=OrderStatus.OPEN, limit=200)
    return trading.get_orders(req)


def cancel_open_stop_orders_for_symbol(symbol: str):
    try:
        orders = list_open_orders()
    except Exception:
        return
    for o in orders:
        try:
            if o.symbol == symbol and str(o.side).lower().endswith("sell") and str(o.order_type).lower().endswith("stop"):
                trading.cancel_order_by_id(o.id)
        except Exception:
            pass


def get_existing_stop_price(symbol: str) -> float | None:
    try:
        orders = list_open_orders()
    except Exception:
        return None
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


# =========================
# Metrics
# =========================
def fetch_and_store_filled_orders(now_et: dt.datetime):
    """
    Pull recently-filled orders and persist in SQLite.
    """
    last_after = kv_get("orders_after_iso", None)

    # Alpaca expects ISO timestamps; keep it conservative
    if last_after:
        try:
            after_dt = dt.datetime.fromisoformat(last_after)
        except Exception:
            after_dt = now_et - dt.timedelta(days=7)
    else:
        after_dt = now_et - dt.timedelta(days=14)

    try:
        req = GetOrdersRequest(status=OrderStatus.FILLED, after=after_dt, limit=200)
        filled = trading.get_orders(req)
    except Exception:
        return  # don't fail trading/reporting because metrics API hiccups

    if not filled:
        kv_set("orders_after_iso", now_et.isoformat())
        return

    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()

    max_filled_at = after_dt
    for o in filled:
        try:
            oid = str(o.id)
            sym = str(o.symbol)
            side = str(o.side).lower()
            qty = safe_float(getattr(o, "filled_qty", None) or getattr(o, "qty", None) or 0.0)
            favg = safe_float(getattr(o, "filled_avg_price", None) or getattr(o, "filled_avg_price", None) or 0.0)

            f_at = getattr(o, "filled_at", None) or getattr(o, "updated_at", None) or getattr(o, "submitted_at", None)
            if f_at is None:
                continue
            # f_at may be datetime already
            if isinstance(f_at, dt.datetime):
                f_dt = f_at
                if f_dt.tzinfo is None:
                    f_dt = f_dt.replace(tzinfo=dt.timezone.utc)
                f_dt = f_dt.astimezone(TZ_ET)
            else:
                f_dt = dt.datetime.fromisoformat(str(f_at).replace("Z", "+00:00")).astimezone(TZ_ET)

            max_filled_at = max(max_filled_at, f_dt)

            cur.execute(
                """
                INSERT OR IGNORE INTO orders_filled(order_id, filled_at, symbol, side, qty, filled_avg_price)
                VALUES(?,?,?,?,?,?)
                """,
                (oid, f_dt.isoformat(), sym, side, qty, favg),
            )
        except Exception:
            continue

    con.commit()
    con.close()

    # advance cursor a bit
    kv_set("orders_after_iso", (max_filled_at + dt.timedelta(seconds=1)).isoformat())


def rebuild_closed_trades():
    """
    Build closed trades from filled orders using FIFO lots.
    Stores each closed chunk as a unique trade_key.
    """
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()

    cur.execute("SELECT order_id, filled_at, symbol, side, qty, filled_avg_price FROM orders_filled ORDER BY filled_at ASC")
    rows = cur.fetchall()

    # FIFO lots: symbol -> list of (entry_time, entry_price, qty_remaining, entry_order_id)
    lots: dict[str, list[tuple[str, float, float, str]]] = {}

    for order_id, filled_at, symbol, side, qty, price in rows:
        symbol = str(symbol)
        side = str(side).lower()
        qty = float(qty or 0.0)
        price = float(price or 0.0)
        if qty <= 0 or price <= 0:
            continue

        if side == "buy":
            lots.setdefault(symbol, []).append((filled_at, price, qty, order_id))
        elif side == "sell":
            if symbol not in lots or not lots[symbol]:
                # sell without known buys (could be manual); ignore for now
                continue
            sell_qty = qty
            while sell_qty > 1e-9 and lots[symbol]:
                entry_time, entry_price, lot_qty, entry_oid = lots[symbol][0]
                take = min(lot_qty, sell_qty)

                pnl = (price - entry_price) * take
                pnl_pct = (price / entry_price - 1.0) if entry_price else 0.0

                trade_key = f"{entry_oid}->{order_id}:{take:.6f}"
                cur.execute(
                    """
                    INSERT OR IGNORE INTO trades_closed(
                        trade_key, symbol, entry_time, exit_time, qty, entry_price, exit_price, pnl, pnl_pct
                    ) VALUES (?,?,?,?,?,?,?,?,?)
                    """,
                    (trade_key, symbol, entry_time, filled_at, take, entry_price, price, pnl, pnl_pct),
                )

                # reduce lot
                lot_qty -= take
                sell_qty -= take
                if lot_qty <= 1e-9:
                    lots[symbol].pop(0)
                else:
                    lots[symbol][0] = (entry_time, entry_price, lot_qty, entry_oid)

    con.commit()
    con.close()


def get_month_metrics(now_et: dt.datetime, start_capital: float, open_pnl: float) -> dict:
    """
    Returns month metrics based on closed trades + equity curve (mark-to-market with open_pnl).
    """
    m0 = month_start_et(now_et)
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()

    cur.execute(
        """
        SELECT symbol, entry_time, exit_time, qty, entry_price, exit_price, pnl, pnl_pct
        FROM trades_closed
        WHERE exit_time >= ?
        ORDER BY exit_time ASC
        """,
        (m0.isoformat(),),
    )
    trades = cur.fetchall()

    pnl_sum = sum(float(t[6] or 0.0) for t in trades)
    wins = sum(1 for t in trades if float(t[6] or 0.0) > 0)
    losses = sum(1 for t in trades if float(t[6] or 0.0) < 0)
    total = len(trades)
    win_rate = (wins / total * 100.0) if total else 0.0

    # equity curve for drawdown (month)
    cur.execute(
        """
        SELECT day, equity
        FROM equity_curve
        WHERE day >= ?
        ORDER BY day ASC
        """,
        (m0.date().isoformat(),),
    )
    eq_rows = cur.fetchall()
    con.close()

    # Build equity series from stored curve, but include today's open_pnl in today's equity snapshot.
    # We'll handle snapshot writing elsewhere; this is read-only.
    max_dd = 0.0
    peak = None
    for day, eq in eq_rows:
        eq = float(eq or 0.0)
        if peak is None or eq > peak:
            peak = eq
        if peak and peak > 0:
            dd = (peak - eq) / peak
            max_dd = max(max_dd, dd)

    # last 5 trades
    last5 = trades[-5:] if trades else []

    return {
        "month_start": m0,
        "trades_total": total,
        "wins": wins,
        "losses": losses,
        "win_rate": win_rate,
        "pnl_sum": pnl_sum,
        "pnl_pct": (pnl_sum / start_capital * 100.0) if start_capital else 0.0,
        "max_dd_pct": max_dd * 100.0,
        "last5": last5,
    }


def write_equity_snapshot(day_et: str, equity_value: float):
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute(
        "INSERT INTO equity_curve(day, equity) VALUES(?,?) ON CONFLICT(day) DO UPDATE SET equity=excluded.equity",
        (day_et, float(equity_value)),
    )
    con.commit()
    con.close()


# =========================
# Main
# =========================
def main():
    ensure_db()

    start_day, end_day, now_et = today_range_et()

    # Pull fills (best-effort) + rebuild trade table
    fetch_and_store_filled_orders(now_et)
    rebuild_closed_trades()

    # ----- Capital base (simulated or real) -----
    acct = trading.get_account()
    if SIMULATED_CAPITAL:
        base_cap = float(SIMULATED_CAPITAL)
        equity = base_cap
        cash = base_cap
        last_equity = base_cap
    else:
        base_cap = safe_float(acct.equity)
        equity = safe_float(acct.equity)
        cash = safe_float(acct.cash)
        last_equity = safe_float(getattr(acct, "last_equity", equity))

    # ----- Data -----
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

    # ----- Market filter -----
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

    # ----- Positions -----
    positions = trading.get_all_positions()
    held_symbol = positions[0].symbol if positions else None

    # ----- Scan candidates -----
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

    candidates.sort(key=lambda x: (x["ret20_pct"] if not math.isnan(x["ret20_pct"]) else -9999), reverse=True)

    top_lines = [
        f"- {i['symbol']}: close {i['close']:.2f} | MA20 {i['ma20']:.2f} | RSI {i['rsi']:.1f} | ret20 {i['ret20_pct']:.1f}%"
        for i in candidates[:3]
    ] or ["- (sin señales hoy)"]

    action_lines = []
    trailing_lines = []

    # ----- Mark-to-market open pnl for metrics (sim) -----
    open_pnl = 0.0
    pos_line = "- (sin posición)"
    if held_symbol:
        pos = positions[0]
        qty = safe_float(pos.qty)
        avg_entry = safe_float(pos.avg_entry_price)
        held_ind = build_indicators(held_symbol)
        if held_ind and qty > 0 and avg_entry > 0:
            close = float(held_ind["close"])
            open_pnl = (close - avg_entry) * qty
            pos_line = f"- {held_symbol}: qty {qty:g} | entry {avg_entry:.2f} | close {close:.2f} | openPnL {fmt_money(open_pnl)} ({(close/avg_entry-1)*100:.1f}%)"
        else:
            pos_line = f"- {held_symbol}: qty {qty:g} | entry {avg_entry:.2f}"

    # ----- Exit + trailing management -----
    if held_symbol:
        pos = positions[0]
        qty = safe_float(pos.qty)
        avg_entry = safe_float(pos.avg_entry_price)
        held_ind = build_indicators(held_symbol)

        if held_ind:
            close = float(held_ind["close"])
            pnl_pct = (close / avg_entry - 1.0) if avg_entry else 0.0

            # Exit if market filter OFF or close < MA20
            if (not market_ok) or (close < held_ind["ma20"]):
                reason = "mercado OFF" if not market_ok else "close < MA20"
                try:
                    cancel_open_stop_orders_for_symbol(held_symbol)
                except Exception:
                    pass

                if qty > 0:
                    sell = MarketOrderRequest(
                        symbol=held_symbol,
                        qty=qty,
                        side=OrderSide.SELL,
                        time_in_force=TimeInForce.DAY,
                    )
                    trading.submit_order(sell)
                    action_lines.append(f"✅ SELL {held_symbol} qty {qty:g} (salida: {reason})")
                else:
                    action_lines.append(f"ℹ️ Tenía {held_symbol} pero qty inválida, no vendí.")
            else:
                action_lines.append(f"ℹ️ Mantengo {held_symbol} (pnl≈{pnl_pct*100:.1f}%, market_ok={market_ok}).")

                # Trailing: if pnl >= +8% then raise stop
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

    # ----- Entry (only if no position) -----
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

                    # OTO: buy market + stop (no take profit)
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

    # ----- Orders today (best-effort) -----
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

    # ----- Metrics (month) -----
    # Sim equity = start_cap + realized_pnl + open_pnl
    # If not simulated, we still show month realized metrics.
    month_metrics = get_month_metrics(now_et, base_cap, open_pnl)

    realized = month_metrics["pnl_sum"]
    sim_equity = base_cap + realized + open_pnl
    # write equity snapshot for drawdown chart
    write_equity_snapshot(now_et.date().isoformat(), sim_equity)

    # ----- Report -----
    stamp = f"{now_et:%Y-%m-%d %H:%M} ET"
    report = []
    report.append(f"📌 Quantito Daily ({stamp})")
    report.append("")
    report.append("🧭 Filtro mercado")
    report.append(f"- {MARKET_FILTER_SYMBOL}: close {mkt['close']:.2f} | {ma_label} {mkt_ma:.2f}")
    report.append(f"- market_ok: {market_ok}")
    report.append("")
    report.append("💼 Simulación")
    report.append(f"- Capital base: ${base_cap:,.2f}" + (" (sim)" if SIMULATED_CAPITAL else ""))
    report.append(f"- Equity estimada: ${sim_equity:,.2f} (realized {fmt_money(realized)} | open {fmt_money(open_pnl)})")
    report.append(f"- Posición: {pos_line}")
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
    report.append("📊 Métricas del mes")
    report.append(f"- Trades: {month_metrics['trades_total']} | Win rate: {month_metrics['win_rate']:.0f}% (W{month_metrics['wins']}/L{month_metrics['losses']})")
    report.append(f"- PnL mes: {fmt_money(month_metrics['pnl_sum'])} ({month_metrics['pnl_pct']:+.2f}%)")
    report.append(f"- Max DD mes: -{month_metrics['max_dd_pct']:.2f}%")

    # last trades
    if month_metrics["last5"]:
        report.append("")
        report.append("🧾 Últimos trades (cerrados)")
        for t in month_metrics["last5"]:
            # t: symbol, entry_time, exit_time, qty, entry_price, exit_price, pnl, pnl_pct
            sym, et, xt, qty, ep, xp, pnl, pp = t
            report.append(f"- {sym} qty {qty:g} | {ep:.2f}→{xp:.2f} | {fmt_money(pnl)} ({pp*100:+.1f}%)")
    else:
        report.append("")
        report.append("🧾 Últimos trades (cerrados)")
        report.append("- (aún no hay trades cerrados este mes)")

    report.append("")
    report.append("🧾 Órdenes hoy")
    report.extend(order_lines)

    telegram_send("\n".join(report))


if __name__ == "__main__":
    main()
