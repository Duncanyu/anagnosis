"""
Quick migration to add email verification columns to existing database.
Run this once to update your local SQLite database.
"""
import sqlite3
import os

DB_PATH = "anagnosis.db"

def migrate():
    if not os.path.exists(DB_PATH):
        print(f"Database {DB_PATH} doesn't exist yet. It will be created with the correct schema on first run.")
        return
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    try:
        # Check if email_verified column exists
        cursor.execute("PRAGMA table_info(users)")
        columns = [row[1] for row in cursor.fetchall()]
        
        if 'email_verified' not in columns:
            print("Adding email_verified column to users table...")
            cursor.execute("ALTER TABLE users ADD COLUMN email_verified VARCHAR(10) DEFAULT 'false' NOT NULL")
            print("✓ Added email_verified column")
        else:
            print("✓ email_verified column already exists")
        
        # Check if email_verification_tokens table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='email_verification_tokens'")
        if not cursor.fetchone():
            print("Creating email_verification_tokens table...")
            cursor.execute("""
                CREATE TABLE email_verification_tokens (
                    id CHAR(36) PRIMARY KEY,
                    user_id CHAR(36) NOT NULL,
                    token VARCHAR(100) NOT NULL UNIQUE,
                    expires_at DATETIME NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id)
                )
            """)
            cursor.execute("CREATE INDEX ix_email_verification_tokens_token ON email_verification_tokens(token)")
            print("✓ Created email_verification_tokens table")
        else:
            print("✓ email_verification_tokens table already exists")
        
        conn.commit()
        print("\n✓ Migration complete!")
        
    except Exception as e:
        print(f"Error during migration: {e}")
        conn.rollback()
    finally:
        conn.close()

if __name__ == "__main__":
    migrate()
