import asyncio

from telegram import Bot
from telegram.error import TelegramError

from stock_ai_agent.src.config import bot_token, chat_id


async def send_alert(message: str, retries: int = 3):
    bot = Bot(token=bot_token)
    for attempt in range(retries):
        try:
            await bot.send_message(chat_id=chat_id, text=message, parse_mode="Markdown")
            return
        except TelegramError as e:
            if attempt < retries - 1:
                await asyncio.sleep(5)
            else:
                print(f"Failed to send Telegram alert after {retries} attempts: {e}")
