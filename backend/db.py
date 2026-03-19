import sqlite3
import uuid
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
            conversation_id TEXT,
            question TEXT,
            answer TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS settings (
            key TEXT PRIMARY KEY,
            value TEXT
        )
    """)

    # Migration: add conversation_id to existing DBs
    try:
        cursor.execute("ALTER TABLE chat_history ADD COLUMN conversation_id TEXT")
    except sqlite3.OperationalError:
        pass  # column already exists

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

def save_chat(username, question, answer, conversation_id=None):
    """Save a Q&A. If conversation_id is None, create a new one. Returns the conversation_id used."""
    if conversation_id is None:
        conversation_id = str(uuid.uuid4())
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO chat_history (username, conversation_id, question, answer) VALUES (?, ?, ?, ?)",
        (username, conversation_id, question, answer)
    )
    conn.commit()
    conn.close()
    return conversation_id


def get_recent_chats(username, conversation_id=None, limit=5):
    """Get recent Q&A for RAG context. If conversation_id given, only that conversation; if None (new chat), return []."""
    if not conversation_id:
        return []
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()
    cursor.execute("""
        SELECT question, answer
        FROM chat_history
        WHERE username=? AND conversation_id=?
        ORDER BY id DESC
        LIMIT ?
    """, (username, conversation_id, limit))
    rows = cursor.fetchall()
    conn.close()
    return rows[::-1]  # oldest → newest


def get_conversations(username):
    """List user's conversations: id, title (first question snippet), updated_at. Newest first."""
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()
    cursor.execute("""
        SELECT conversation_id, question, timestamp
        FROM chat_history
        WHERE username=? AND conversation_id IS NOT NULL
        ORDER BY id ASC
    """, (username,))
    rows = cursor.fetchall()
    conn.close()
    # Build: conv_id -> { first_question, last_updated }
    by_cid = {}
    for cid, question, ts in rows:
        if cid not in by_cid:
            by_cid[cid] = {"title": (question or "New chat").strip()[:60], "updated_at": ts}
        else:
            by_cid[cid]["updated_at"] = ts
    out = [{"id": cid, "title": (d["title"][:60] + ("..." if len(d["title"]) >= 60 else "")) or "New chat", "updated_at": d["updated_at"]}
             for cid, d in by_cid.items()]
    out.sort(key=lambda x: x["updated_at"] or "", reverse=True)
    return out[:50]


def get_conversation_messages(username, conversation_id):
    """Get all Q&A for a conversation. Returns list of {question, answer} in order."""
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()
    cursor.execute("""
        SELECT question, answer
        FROM chat_history
        WHERE username=? AND conversation_id=?
        ORDER BY id ASC
    """, (username, conversation_id))
    rows = cursor.fetchall()
    conn.close()
    return [{"question": q, "answer": a} for q, a in rows]

def get_admin_stats():
    """Get usage statistics for the admin dashboard"""
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()
    
    # Get user query counts
    cursor.execute("""
        SELECT u.username, u.role, COUNT(c.id) as query_count
        FROM users u
        LEFT JOIN chat_history c ON u.username = c.username
        GROUP BY u.username
    """)
    stats = cursor.fetchall()
    conn.close()
    
    return [
        {"username": row[0], "role": row[1], "queries": row[2]}
        for row in stats
    ]

def delete_user(username):
    """Delete a user and their full chat history"""
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()
    try:
        cursor.execute("DELETE FROM chat_history WHERE username=?", (username,))
        cursor.execute("DELETE FROM users WHERE username=?", (username,))
        conn.commit()
    finally:
        conn.close()

def get_query_time_stats():
    """Get number of queries per day for the last 30 days"""
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()
    cursor.execute("""
        SELECT date(timestamp) as day, COUNT(*) as count
        FROM chat_history
        GROUP BY day
        ORDER BY day DESC
        LIMIT 30
    """)
    rows = cursor.fetchall()
    conn.close()
    return [{"day": r[0], "count": r[1]} for r in rows[::-1]]

def get_settings():
    """Get all system settings"""
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()
    cursor.execute("SELECT key, value FROM settings")
    rows = cursor.fetchall()
    conn.close()
    # Default settings if empty
    defaults = {"maintenance_mode": "false", "allow_signup": "true", "max_upload_size": "20"}
    settings = {r[0]: r[1] for r in rows}
    return {**defaults, **settings}

def update_setting(key, value):
    """Update or insert a system setting"""
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()
    cursor.execute("INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)", (key, str(value)))
    conn.commit()
    conn.close()

