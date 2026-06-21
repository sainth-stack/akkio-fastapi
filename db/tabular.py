from __future__ import annotations

import pickle
from datetime import datetime

import pandas as pd
import psycopg2


class TabularDataMixin:
    def create_table(self):
        """Creates akio_data_fastapi for tabular multi-model uploads."""
        self.ensure_connection()
        with self.connection.cursor() as cursor:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS akio_data_fastapi(
                    id SERIAL PRIMARY KEY,
                    email VARCHAR(255) NOT NULL,
                    name VARCHAR(255),
                    lastupdate TIMESTAMP,
                    datecreated TIMESTAMP,
                    fileobj BYTEA,
                    type VARCHAR(50),
                    subtype VARCHAR(50),
                    rawfile BYTEA
                )
            """)
            for col, typ in (
                ("type", "VARCHAR(50)"),
                ("subtype", "VARCHAR(50)"),
                ("rawfile", "BYTEA"),
            ):
                try:
                    cursor.execute(
                        f"ALTER TABLE akio_data_fastapi ADD COLUMN IF NOT EXISTS {col} {typ}"
                    )
                except Exception:
                    pass

    def _ensure_type_columns(self):
        self.ensure_connection()
        with self.connection.cursor() as cursor:
            for col, typ in (
                ("type", "VARCHAR(50)"),
                ("subtype", "VARCHAR(50)"),
                ("rawfile", "BYTEA"),
            ):
                try:
                    cursor.execute(
                        f"ALTER TABLE akio_data_fastapi ADD COLUMN IF NOT EXISTS {col} {typ}"
                    )
                except Exception:
                    pass

    def insert_or_update(
        self,
        email,
        data,
        tb_name,
        data_type: str | None = None,
        data_subtype: str | None = None,
        raw_bytes: bytes | None = None,
    ):
        self.ensure_connection()
        self._ensure_type_columns()
        tb_name_clean = self._clean_name(tb_name)
        blob_data = pickle.dumps(data)
        with self.connection.cursor() as cursor:
            cursor.execute(
                "SELECT id FROM akio_data_fastapi WHERE email = %s AND name = %s",
                (email, tb_name_clean),
            )
            existing = cursor.fetchone()
            if existing:
                cursor.execute(
                    """
                    UPDATE akio_data_fastapi
                    SET lastupdate = %s, fileobj = %s,
                        rawfile = COALESCE(%s, rawfile),
                        type = COALESCE(%s, type),
                        subtype = COALESCE(%s, subtype)
                    WHERE email = %s AND name = %s
                    """,
                    (
                        datetime.now(),
                        psycopg2.Binary(blob_data),
                        psycopg2.Binary(raw_bytes) if raw_bytes is not None else None,
                        data_type,
                        data_subtype,
                        email,
                        tb_name_clean,
                    ),
                )
                return "updated"
            cursor.execute(
                """
                INSERT INTO akio_data_fastapi (
                    email, name, lastupdate, datecreated, fileobj, rawfile, type, subtype
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    email,
                    tb_name_clean,
                    datetime.now(),
                    datetime.now(),
                    psycopg2.Binary(blob_data),
                    psycopg2.Binary(raw_bytes) if raw_bytes is not None else None,
                    data_type,
                    data_subtype,
                ),
            )
            return "inserted"

    def get_table_data(self, table_name):
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    clean_name = self._clean_name(table_name)
                    cursor.execute(
                        "SELECT fileobj FROM akio_data_fastapi WHERE name = %s LIMIT 1",
                        (clean_name,),
                    )
                    row = cursor.fetchone()
                    if not row or row[0] is None:
                        return pd.DataFrame()
                    table_data = pickle.loads(bytes(row[0]))
                    if isinstance(table_data, pd.DataFrame):
                        return table_data
                    return pd.DataFrame(table_data)
        except Exception as e:
            print(f"Error getting table data for {table_name}: {e}")
            raise
