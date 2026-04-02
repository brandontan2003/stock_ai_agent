import logging
import os
import sys

from dotenv import load_dotenv


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stdout,
    )


setup_logging()
load_dotenv()

stock_symbol = os.getenv("STOCK_SYMBOL", "^GSPC")
bot_token = os.getenv("BOT_TOKEN")
chat_id = os.getenv("TELEGRAM_CHAT_ID")
poll_interval = int(os.getenv("POLL_INTERVAL", "300"))

# Anthropic API Key
api_key = os.getenv("ANTHROPIC_API_KEY")

# How many recent signals to include as context
history_window = int(os.getenv("HISTORY_WINDOW", "10"))
