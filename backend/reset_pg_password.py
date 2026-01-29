import os
import psycopg

DB_HOST = os.getenv("DB_HOST", "").strip()
DB_PORT = os.getenv("DB_PORT", "").strip()
DB_NAME = os.getenv("DB_NAME", "postgres").strip()
DB_USER = os.getenv("DB_USER", "").strip()
DB_PASSWORD = os.getenv("DB_PASSWORD", "").strip()
NEW_PASSWORD = os.getenv("DB_NEW_PASSWORD", "").strip()

missing = [k for k, v in {
    "DB_HOST": DB_HOST,
    "DB_PORT": DB_PORT,
    "DB_USER": DB_USER,
    "DB_NEW_PASSWORD": NEW_PASSWORD,
}.items() if not v]

if missing:
    raise ValueError(f"缺少环境变量：{', '.join(missing)}")

conn_parts = [
    f"host={DB_HOST}",
    f"port={DB_PORT}",
    f"dbname={DB_NAME}",
    f"user={DB_USER}",
]
if DB_PASSWORD:
    conn_parts.append(f"password={DB_PASSWORD}")

conn = psycopg.connect(" ".join(conn_parts))

with conn.cursor() as cur:
    # 这里不要用 %s 参数绑定，直接写字面量
    cur.execute(f"ALTER USER {DB_USER} WITH PASSWORD '{NEW_PASSWORD}';")

conn.commit()
conn.close()
print("password reset OK")
