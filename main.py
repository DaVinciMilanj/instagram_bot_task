import os
import re
import sqlite3
import traceback
from pathlib import Path
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
HF_MODEL = os.getenv("HF_MODEL", "meta-llama/Llama-3.1-8B-Instruct:novita")

client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=HF_TOKEN
)

DB_PATH = Path("db") / "app_data.sqlite"
if not DB_PATH.exists():
    raise RuntimeError("Database not found. Run db_init.py first!")

app = FastAPI(title="Salona Instagram Bot (RAG + LLM)")


@app.get("/")
def home():
    return {"message": "🤖 RAG Bot is running successfully!"}


class DMIn(BaseModel):
    sender_id: str
    message_id: str
    text: str


def connect_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def tokenize(text: str) -> List[str]:
    t = re.sub(r"[^\w\u0600-\u06FF\s]", " ", text.lower())
    return [tok for tok in t.split() if len(tok) > 1]


def rag_retrieve(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    tokens = tokenize(query)
    if not tokens:
        return []
    conn = connect_db()
    cur = conn.cursor()
    cur.execute("SELECT id, name, description, price FROM products")
    rows = cur.fetchall()
    scored = []
    for r in rows:
        text = f"{r['name']} {r['description']}".lower()
        score = sum(text.count(tok) for tok in tokens)
        if score > 0:
            scored.append((score, dict(r)))
    scored.sort(key=lambda x: x[0], reverse=True)
    conn.close()
    return [item for _, item in scored[:top_k]]


def call_openrouter_model(prompt: str) -> str:
    if not HF_TOKEN:
        return "توکن HF تنظیم نشده است."
    try:
        completion = client.chat.completions.create(
            model=HF_MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        traceback.print_exc()
        return f"خطا در تماس با OpenRouter: {str(e)}"


def build_prompt(user_text: str, retrieved: List[Dict[str, Any]]) -> str:
    intro = (
        "تو یک دستیار فروش فارسی هستی. "
        "اطلاعات زیر از دیتابیس بازیابی شده. "
        "فقط به فارسی و مختصر پاسخ بده.\n\n"
    )
    db_info = "اطلاعات محصولات:\n"
    for p in retrieved:
        db_info += f"- {p['name']}: {p['description']}، قیمت: {int(p['price'])} تومان\n"
    return f"{intro}{db_info}\nسوال کاربر: {user_text}\nپاسخ فارسی:"


@app.post("/simulate_dm")
def simulate_dm(dm: DMIn):
    try:
        retrieved = rag_retrieve(dm.text)
        prompt = build_prompt(dm.text, retrieved)
        reply = call_openrouter_model(prompt)
        return {"reply": reply}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
