import os
from dotenv import load_dotenv

load_dotenv()

stock_symbol = os.getenv("STOCK_SYMBOL", "^GSPC")
bot_token = os.getenv("BOT_TOKEN")
chat_id = os.getenv("TELEGRAM_CHAT_ID")
poll_interval = int(os.getenv("POLL_INTERVAL", "300"))
