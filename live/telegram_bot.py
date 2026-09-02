"""
ارسال پیام به تلگرام + پشتیبانی از «تاپیک‌های جدا برای هر استراتژی».

اگر چت مقصد یک گروه از نوع Forum (تاپیک‌دار) باشد و ربات ادمین آن با
دسترسی Manage Topics باشد، این ماژول برای هر استراتژی یک تاپیک مجزا
می‌سازد (یک‌بار) و همه پیام‌های آن استراتژی (سیگنال باز شدن، بسته شدن
با سود/ضرر، خلاصه ماهانه) را داخل همان تاپیک می‌فرستد. این‌طوری توی
اپ تلگرام هر استراتژی یک ساب‌چت جدا دارد و می‌توانی جدا اسکرول/تحلیل کنی.

اگر چت مقصد Forum نباشد (مثلاً یک چت خصوصی معمولی با ربات)، به‌صورت
خودکار fallback می‌زند: پیام‌ها بدون تاپیک ولی با تگ واضح استراتژی در
همان چت عادی ارسال می‌شوند - هیچ‌چیز خراب نمی‌شود.

راه‌اندازی تاپیک‌ها (اختیاری ولی توصیه‌شده):
۱. یک گروه تلگرام بساز (نه کانال، نه چت خصوصی با ربات).
۲. در تنظیمات گروه، "Topics" را فعال کن (گروه را Forum کن).
۳. ربات را به گروه اضافه کن و ادمینش کن با دسترسی "Manage Topics".
۴. chat_id همین گروه را در config.yaml بگذار.
"""
from __future__ import annotations
import json
import os
import requests
from typing import Optional


TOPICS_CACHE_PATH = "results/live/telegram_topics.json"


def send_telegram_message(bot_token: str, chat_id: str, text: str,
                           message_thread_id: Optional[int] = None) -> bool:
    if not bot_token or "PUT_YOUR" in bot_token:
        print("[telegram] توکن تنظیم نشده - پیام فقط چاپ می‌شود:\n", text)
        return False
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }
    if message_thread_id is not None:
        payload["message_thread_id"] = message_thread_id
    try:
        resp = requests.post(url, data=payload, timeout=15)
        if resp.status_code == 400 and message_thread_id is not None:
            # تاپیک احتمالاً پاک شده یا نامعتبر است؛ به‌صورت معمولی (بدون تاپیک) دوباره بفرست
            print("[telegram] ارسال به تاپیک ناموفق بود، ارسال معمولی جایگزین شد.")
            payload.pop("message_thread_id")
            resp = requests.post(url, data=payload, timeout=15)
        resp.raise_for_status()
        return True
    except Exception as e:
        print(f"[telegram] خطا در ارسال پیام: {e}")
        return False


def _load_topics_cache() -> dict:
    if not os.path.exists(TOPICS_CACHE_PATH):
        return {}
    try:
        with open(TOPICS_CACHE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_topics_cache(cache: dict):
    os.makedirs(os.path.dirname(TOPICS_CACHE_PATH), exist_ok=True)
    with open(TOPICS_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def get_or_create_topic(bot_token: str, chat_id: str, strategy_name: str,
                         use_topics: bool = True) -> Optional[int]:
    """
    برای یک استراتژی، شناسه تاپیک تلگرام را برمی‌گرداند - از کش می‌خواند
    یا (اگر نبود) یک تاپیک جدید در گروه می‌سازد. اگر use_topics=False یا
    گروه Forum نباشد، None برمی‌گرداند (یعنی پیام معمولی/بدون تاپیک بفرست).
    """
    if not use_topics or not bot_token or "PUT_YOUR" in bot_token:
        return None

    cache = _load_topics_cache()
    key = f"{chat_id}::{strategy_name}"

    if key in cache:
        return cache[key] if cache[key] != "unsupported" else None

    url = f"https://api.telegram.org/bot{bot_token}/createForumTopic"
    try:
        resp = requests.post(url, data={"chat_id": chat_id, "name": strategy_name}, timeout=15)
        data = resp.json()
        if data.get("ok"):
            thread_id = data["result"]["message_thread_id"]
            cache[key] = thread_id
            _save_topics_cache(cache)
            return thread_id
        else:
            # چت Forum نیست یا ربات دسترسی ندارد -> از این به بعد دیگر تلاش نکن
            print(f"[telegram] ساخت تاپیک برای {strategy_name} ممکن نشد: {data.get('description')}")
            cache[key] = "unsupported"
            _save_topics_cache(cache)
            return None
    except Exception as e:
        print(f"[telegram] خطا در ساخت تاپیک: {e}")
        return None


def format_signal_message(strategy_name: str, category: str, timeframe: str, symbol: str, signal) -> str:
    emoji = "🟢" if signal.side == "long" else "🔴"
    return (
        f"{emoji} <b>سیگنال جدید ({'خرید' if signal.side == 'long' else 'فروش'})</b>\n"
        f"نماد: <b>{symbol}</b> | تایم‌فریم: <b>{timeframe}</b>\n"
        f"استراتژی: <code>{strategy_name}</code> [{category}]\n"
        f"ورود: <code>{signal.entry:.6g}</code>\n"
        f"حد ضرر: <code>{signal.stop_loss:.6g}</code>\n"
        f"حد سود: <code>{signal.take_profit:.6g}</code>\n"
        f"R:R تقریبی: {signal.rr}\n"
        f"دلیل: {signal.reason}\n"
        f"⚠️ این یک پوزیشن فرضی (پیپر‌تریدینگ) است، توصیه مالی نیست."
    )


def format_close_message(position) -> str:
    emoji = "✅" if position.status == "win" else "❌"
    return (
        f"{emoji} <b>پوزیشن بسته شد ({'سود' if position.status == 'win' else 'ضرر'})</b>\n"
        f"نماد: <b>{position.symbol}</b> | تایم‌فریم: <b>{position.timeframe}</b>\n"
        f"استراتژی: <code>{position.strategy}</code> [{position.category}]\n"
        f"ورود: <code>{position.entry_price:.6g}</code> | خروج: <code>{position.exit_price:.6g}</code>\n"
        f"PnL: <b>{position.pnl_pct}%</b> ({position.pnl_usdt} USDT)"
    )
