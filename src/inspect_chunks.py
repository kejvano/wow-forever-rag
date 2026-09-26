import sys

from db import connect

term = " ".join(sys.argv[1:])
rows = connect().execute(
    """
    SELECT a.title, c.position, c.text
    FROM chunks c JOIN articles a ON a.id = c.article_id
    WHERE c.text LIKE ?
    ORDER BY a.title, c.position
    """,
    (f"%{term}%",),
).fetchall()
for title, position, text in rows:
    print(f"=== {title} [chunk {position}] ===\n{text}\n")
print(f"{len(rows)} chunks contain '{term}'")