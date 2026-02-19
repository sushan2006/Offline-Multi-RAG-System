import sqlite3
from passlib.context import CryptContext

# Use passlib to hash & verify passwords
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def init_db():
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            password TEXT,
            role TEXT
        )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS chat_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT,
        question TEXT,
        answer TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
""")


    conn.commit()
    conn.close()


def create_user(username, password, role):
    """Create a user and store a hashed password. Raises ValueError on duplicate username."""
    hashed = pwd_context.hash(password)
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()

    try:
        cursor.execute(
            "INSERT INTO users (username, password, role) VALUES (?, ?, ?)",
            (username, hashed, role)
        )
        conn.commit()
    except sqlite3.IntegrityError:
        raise ValueError("Username already exists")
    finally:
        conn.close()


def get_user(username, password):
    """Return user row if username/password match. Supports migrating plaintext passwords on first login."""
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()

    cursor.execute(
        "SELECT * FROM users WHERE username=?",
        (username,)
    )

    user = cursor.fetchone()

    if not user:
        conn.close()
        return None

    stored_password = user[2]  # password column

    # First try verifying as a hashed password
    try:
        if pwd_context.verify(password, stored_password):
            conn.close()
            return user
    except Exception:
        # stored_password may be plaintext (older DB) — fallback to plain comparison
        if stored_password == password:
            # upgrade stored password to a hash
            new_hash = pwd_context.hash(password)
            cursor.execute("UPDATE users SET password=? WHERE username=?", (new_hash, username))
            conn.commit()
            conn.close()
            return user

    conn.close()
    return None

def save_chat(username, question, answer):
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()

    cursor.execute(
        "INSERT INTO chat_history (username, question, answer) VALUES (?, ?, ?)",
        (username, question, answer)
    )

    conn.commit()
    conn.close()


def get_recent_chats(username, limit=5):
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()

    cursor.execute("""
        SELECT question, answer
        FROM chat_history
        WHERE username=?
        ORDER BY id DESC
        LIMIT ?
    """, (username, limit))

    rows = cursor.fetchall()
    conn.close()

    return rows[::-1]  # return oldest → newest

