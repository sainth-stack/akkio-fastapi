from __future__ import annotations


class LLMSettingsMixin:
    def create_llm_settings_table(self):
        self.ensure_connection()
        with self.connection.cursor() as cursor:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS llm_settings (
                    id SERIAL PRIMARY KEY,
                    email VARCHAR(255) UNIQUE NOT NULL,
                    provider VARCHAR(50) DEFAULT 'openai',
                    api_key TEXT,
                    model_name VARCHAR(255),
                    created_at TIMESTAMP DEFAULT NOW(),
                    updated_at TIMESTAMP DEFAULT NOW()
                )
            """)
            try:
                cursor.execute("""
                    ALTER TABLE llm_settings
                    ADD COLUMN IF NOT EXISTS provider VARCHAR(50) DEFAULT 'openai'
                """)
            except Exception:
                pass

    def save_llm_settings(
        self, email: str, provider: str | None = None, api_key: str | None = None, model_name: str | None = None
    ):
        self.ensure_connection()
        with self.connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO llm_settings (email, provider, api_key, model_name, created_at, updated_at)
                VALUES (%s, %s, %s, %s, NOW(), NOW())
                ON CONFLICT (email) DO UPDATE SET
                    provider = COALESCE(EXCLUDED.provider, llm_settings.provider),
                    api_key = COALESCE(EXCLUDED.api_key, llm_settings.api_key),
                    model_name = COALESCE(EXCLUDED.model_name, llm_settings.model_name),
                    updated_at = NOW()
                """,
                (email, provider, api_key, model_name),
            )

    def get_llm_settings(self, email: str):
        self.ensure_connection()
        with self.connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT email, provider, api_key, model_name, created_at, updated_at
                FROM llm_settings WHERE email = %s
                """,
                (email,),
            )
            row = cursor.fetchone()
            if not row:
                return None
            keys = ["email", "provider", "api_key", "model_name", "created_at", "updated_at"]
            return dict(zip(keys, row))

    def delete_llm_settings(self, email: str):
        self.ensure_connection()
        with self.connection.cursor() as cursor:
            cursor.execute("DELETE FROM llm_settings WHERE email = %s", (email,))
            return f"{cursor.rowcount} LLM settings deleted"
