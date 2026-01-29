import psycopg

conn = psycopg.connect("host=127.0.0.1 port=5432 dbname=postgres user=postgres")

with conn.cursor() as cur:
    # 这里不要用 %s 参数绑定，直接写字面量
    cur.execute("ALTER USER postgres WITH PASSWORD 'postgres';")

conn.commit()
conn.close()
print("password reset OK")