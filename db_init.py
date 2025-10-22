import sqlite3
from pathlib import Path

DB_PATH = Path("db") / "app_data.sqlite"
DB_PATH.parent.mkdir(exist_ok=True)

conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

cur.execute("""
CREATE TABLE IF NOT EXISTS products (
    id INTEGER PRIMARY KEY,
    name TEXT,
    description TEXT,
    price REAL
)
""")

# حذف داده‌های قدیمی
cur.execute("DELETE FROM products")

# افزودن 100 محصول تستی
for i in range(1, 101):
    cur.execute(
        "INSERT INTO products (name, description, price) VALUES (?, ?, ?)",
        (f"محصول {i}", f"توضیح محصول شماره {i}", 1000000 + i*5000)
    )

conn.commit()
conn.close()
print("created")