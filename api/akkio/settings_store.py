import json
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from database import PostgresDatabase
from pydantic import BaseModel

class WorkspaceSettings(BaseModel):
    email: str
    workspace_name: Optional[str] = None
    icon_url: Optional[str] = None
    theme_color: Optional[str] = None
    plan_id: Optional[str] = "free"

class Transaction(BaseModel):
    id: int
    email: str
    source: str
    type: str
    amount: float
    details: Optional[str]
    created_at: datetime

def _ensure_settings_tables(db: PostgresDatabase) -> None:
    db.ensure_connection()
    with db.connection.cursor() as cursor:
        # Workspace Settings Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS workspace_settings (
                email VARCHAR(255) PRIMARY KEY,
                workspace_name VARCHAR(255),
                icon_url TEXT,
                theme_color VARCHAR(50),
                plan_id VARCHAR(50) DEFAULT 'free',
                created_at TIMESTAMP DEFAULT NOW(),
                updated_at TIMESTAMP DEFAULT NOW()
            )
        """)
        
        # Credit Transactions Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS credit_transactions (
                id SERIAL PRIMARY KEY,
                email VARCHAR(255) NOT NULL,
                source VARCHAR(50),
                type VARCHAR(50),
                amount DECIMAL(10, 2),
                details TEXT,
                created_at TIMESTAMP DEFAULT NOW()
            )
        """)
        
        # Index on email for transactions
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_credit_transactions_email ON credit_transactions(email)
        """)
        
        # Index on created_at for transactions
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_credit_transactions_created_at ON credit_transactions(created_at)
        """)

def get_workspace_settings(db: PostgresDatabase, email: str) -> Dict[str, Any]:
    _ensure_settings_tables(db)
    with db.connection.cursor() as cursor:
        cursor.execute("""
            SELECT workspace_name, icon_url, theme_color, plan_id 
            FROM workspace_settings 
            WHERE email = %s
        """, (email,))
        row = cursor.fetchone()
        
        if row:
            return {
                "workspace_name": row[0],
                "icon_url": row[1],
                "theme_color": row[2],
                "plan_id": row[3] or "free"
            }
        else:
            # Return defaults
            return {
                "workspace_name": email.split('@')[0], # Default to username part of email
                "icon_url": "",
                "theme_color": "#6366f1",
                "plan_id": "free"
            }

def update_workspace_settings(db: PostgresDatabase, email: str, settings: WorkspaceSettings) -> Dict[str, Any]:
    _ensure_settings_tables(db)
    with db.connection.cursor() as cursor:
        cursor.execute("""
            INSERT INTO workspace_settings (email, workspace_name, icon_url, theme_color, updated_at)
            VALUES (%s, %s, %s, %s, NOW())
            ON CONFLICT (email) DO UPDATE SET
                workspace_name = EXCLUDED.workspace_name,
                icon_url = EXCLUDED.icon_url,
                theme_color = EXCLUDED.theme_color,
                updated_at = NOW()
            RETURNING workspace_name, icon_url, theme_color, plan_id
        """, (email, settings.workspace_name, settings.icon_url, settings.theme_color))
        row = cursor.fetchone()
        return {
            "workspace_name": row[0],
            "icon_url": row[1],
            "theme_color": row[2],
            "plan_id": row[3]
        }

def update_plan(db: PostgresDatabase, email: str, plan_id: str) -> str:
    _ensure_settings_tables(db)
    with db.connection.cursor() as cursor:
        cursor.execute("""
            INSERT INTO workspace_settings (email, plan_id, updated_at)
            VALUES (%s, %s, NOW())
            ON CONFLICT (email) DO UPDATE SET
                plan_id = EXCLUDED.plan_id,
                updated_at = NOW()
        """, (email, plan_id))
        return plan_id

def log_transaction(db: PostgresDatabase, email: str, source: str, type: str, amount: float, details: str = None):
    _ensure_settings_tables(db)
    with db.connection.cursor() as cursor:
        cursor.execute("""
            INSERT INTO credit_transactions (email, source, type, amount, details, created_at)
            VALUES (%s, %s, %s, %s, %s, NOW())
        """, (email, source, type, amount, details))

def get_transactions(db: PostgresDatabase, email: str, limit: int = 10) -> List[Dict[str, Any]]:
    _ensure_settings_tables(db)
    with db.connection.cursor() as cursor:
        cursor.execute("""
            SELECT created_at, source, type, amount, details
            FROM credit_transactions
            WHERE email = %s
            ORDER BY created_at DESC
            LIMIT %s
        """, (email, limit))
        rows = cursor.fetchall()
        
        return [
            {
                "date": row[0].strftime("%b %d, %Y"),
                "time": row[0].strftime("%H:%M:%S"),
                "user": "You", # Placeholder for now, multi-user support later
                "source": row[1],
                "type": row[2],
                "credits": float(row[3]),
                "details": row[4] or "-"
            }
            for row in rows
        ]

def get_daily_usage(db: PostgresDatabase, email: str, days: int = 14) -> List[Dict[str, Any]]:
    _ensure_settings_tables(db)
    with db.connection.cursor() as cursor:
        # Get usage (negative amounts) aggregated by day
        cursor.execute("""
            SELECT DATE(created_at) as day, SUM(amount) as daily_credits
            FROM credit_transactions
            WHERE email = %s 
              AND amount < 0 
              AND created_at >= NOW() - INTERVAL '%s days'
            GROUP BY DATE(created_at)
            ORDER BY day ASC
        """, (email, days))
        rows = cursor.fetchall()
        
        # Fill in missing days
        usage_map = {row[0].strftime("%Y-%m-%d"): float(row[1]) for row in rows}
        result = []
        today = datetime.now().date()
        for i in range(days):
            date = today - timedelta(days=days-1-i)
            date_str = date.strftime("%Y-%m-%d")
            display_date = date.strftime("%b %d")
            result.append({
                "date": display_date,
                "credits": usage_map.get(date_str, 0)
            })
            
        return result

def get_analytics_summary(db: PostgresDatabase, email: str) -> Dict[str, Any]:
    _ensure_settings_tables(db)
    with db.connection.cursor() as cursor:
        # Total Credits Used (Sum of negative transactions)
        cursor.execute("""
            SELECT SUM(amount), COUNT(*)
            FROM credit_transactions
            WHERE email = %s AND amount < 0
        """, (email,))
        row = cursor.fetchone()
        total_used = float(row[0] or 0)
        tx_count = row[1]
        
        # Daily Average (over last 30 days)
        cursor.execute("""
            SELECT SUM(amount)
            FROM credit_transactions
            WHERE email = %s 
              AND amount < 0 
              AND created_at >= NOW() - INTERVAL '30 days'
        """, (email,))
        month_used = float(cursor.fetchone()[0] or 0)
        daily_avg = month_used / 30.0 if month_used else 0.0
        
        # Active Users (count distinct emails in last 30 days - for now just 1 or count distinct if we had user_id)
        # Since we query by email, it's always 1 user context unless we have team logic.
        # Placeholder: 1
        
    return {
        "total_used": total_used,
        "transaction_count": tx_count,
        "daily_average": round(daily_avg, 2),
        "active_users": 1
    }


