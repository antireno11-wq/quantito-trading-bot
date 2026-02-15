import os
import datetime as dt
from zoneinfo import ZoneInfo
import requests
from alpaca.trading.client import TradingClient

TZ_ET = ZoneInfo("America/New_York")

ALPACA_KEY = os.environ["ALPACA_KEY"]
ALPACA_SECRET = os.environ["ALPACA_SECRET"]
ALPACA_BASE_URL = os.environ["ALPACA_BASE_URL"]

TG_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
TG_CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]

trading = TradingClient(ALPACA_KEY, ALPACA_SECRET, paper=("paper" in ALPACA_BASE_URL))


def telegram_send(text: str):
    url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
    r = requests.post(url, json={"chat_id": TG_CHAT_ID, "text": text}, timeout=20)
    r.raise_for_status()


def main():
    now_et = dt.datetime.now(TZ_ET)
    acct = trading.get_account()

    msg = (
        f"📌 Quantito Daily Report ({now_et:%Y-%m-%d %H:%M} ET)\n"
        f"Equity: ${float(acct.equity):,.2f}\n"
        f"Cash: ${float(acct.cash):,.2f}\n"
        f"Status: OK"
    )

    telegram_send(msg)


if __name__ == "__main__":
    main()
