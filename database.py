import psycopg2
import pandas as pd
import pickle
import base64
import psycopg2.errors
from datetime import datetime
from fastapi import HTTPException

class PostgresDatabase:
    def __init__(self):
        self.connection = None

    def create_connection(self, user, password, database, host, port=5432, return_tables_info: bool = False):
        try:
            self.connection = psycopg2.connect(
                database=database,
                user=user,
                password=password,
                host=host,
                port=port
            )
            self.connection.autocommit = True
            print("Database connection established.")
            if return_tables_info:
                return self.get_tables_info()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"DB Connection Error: {e}")

    def ensure_connection(self):
        try:
            if self.connection is None or self.connection.closed:
                print("Reconnecting to the database...")
                self.create_connection(
                    user=PGUSER,
                    password=PGPASSWORD,
                    database=PGDATABASE,
                    host=PGHOST
                )
        except Exception as e:
            print(f"Error ensuring connection: {e}")
            raise

    def create_table(self):
        """Creates akio_data_fastapi with no unique constraints on email, allowing duplicates."""
        self.ensure_connection()
        with self.connection.cursor() as cursor:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS akio_data_fastapi(
                    id SERIAL PRIMARY KEY,
                    email VARCHAR(255) NOT NULL,
                    name VARCHAR(255),
                    lastupdate TIMESTAMP,
                    datecreated TIMESTAMP,
                    fileobj BYTEA
                )
            """)

    def create_reports_table(self):
        """Creates reports_fastapi table with url field instead of image_bytes."""
        self.ensure_connection()
        with self.connection.cursor() as cursor:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS reports_fastapi (
                    id SERIAL PRIMARY KEY,
                    email VARCHAR(255),
                    url TEXT,
                    created_at TIMESTAMP DEFAULT NOW(),
                    updated_at TIMESTAMP DEFAULT NOW()
                )
            """)

    def insert_or_update(self, email, data, tb_name):
        """Insert or update a row in akio_data_fastapi. Allows multiple rows per email."""
        self.ensure_connection()
        tb_name_clean = tb_name.split('.')[0]
        blob_data = pickle.dumps(data)
        with self.connection.cursor() as cursor:
            cursor.execute("SELECT id FROM akio_data_fastapi WHERE email = %s AND name = %s", (email, tb_name_clean))
            existing = cursor.fetchone()
            if existing:
                cursor.execute("""
                    UPDATE akio_data_fastapi
                    SET lastupdate = %s, fileobj = %s
                    WHERE email = %s AND name = %s
                """, (datetime.now(), psycopg2.Binary(blob_data), email, tb_name_clean))
                return "updated"
            else:
                cursor.execute("""
                    INSERT INTO akio_data_fastapi (email, name, lastupdate, datecreated, fileobj)
                    VALUES (%s, %s, %s, %s, %s)
                """, (email, tb_name_clean, datetime.now(), datetime.now(), psycopg2.Binary(blob_data)))
                return "inserted"

    def read(self):
        self.ensure_connection()
        with self.connection.cursor() as cursor:
            cursor.execute("SELECT * FROM akio_data_fastapi")
            rows = cursor.fetchall()
        cols = ['id', 'email', 'name', 'lastupdate', 'datecreated', 'fileobj']
        return pd.DataFrame(rows, columns=cols)

    def get_tables_info(self):
        """Returns table info, gracefully handles missing table."""
        try:
            df = self.read()
            if df.empty:
                return {}
            df['lastupdate'] = df['lastupdate'].apply(lambda x: x.isoformat() if pd.notnull(x) else None)
            df['datecreated'] = df['datecreated'].apply(lambda x: x.isoformat() if pd.notnull(x) else None)
            return df.iloc[:, :-1].to_dict(orient="records")
        except psycopg2.errors.UndefinedTable:
            print("Table does not exist, returning empty info.")
            return {}
        except Exception as e:
            print(f"Unexpected error in get_tables_info: {e}")
            return {}

    def get_user_tables(self, user):
        df = self.read()
        return list(df[df['email'] == user]['name']) if not df.empty else []

    def get_table_data(self, table_name):
        df = self.read()
        row = df[df['name'] == table_name]["fileobj"]
        if row.empty:
            raise HTTPException(status_code=404, detail="Table not found")
        return pickle.loads(row.values[0].tobytes())

    def delete_tables_data(self, email, table_names):
        if not table_names:
            return "No table names provided"
        self.ensure_connection()
        with self.connection.cursor() as cursor:
            placeholders = ','.join(['%s'] * len(table_names))
            cursor.execute(
                f"DELETE FROM akio_data_fastapi WHERE email = %s AND name IN ({placeholders})",
                [email] + table_names
            )
            return f"{cursor.rowcount} records deleted"

    def delete_all_tables_data(self, email):
        self.ensure_connection()
        with self.connection.cursor() as cursor:
            cursor.execute("DELETE FROM akio_data_fastapi WHERE email = %s", (email,))
            return f"{cursor.rowcount} records deleted"

    def insert_report(self, email, url):
        self.ensure_connection()
        with self.connection.cursor() as cursor:
            cursor.execute("""
                INSERT INTO reports_fastapi (email, url)
                VALUES (%s, %s)
                RETURNING id, email, url, created_at
            """, (email, url))
            result = cursor.fetchone()
            self.connection.commit()  # commit after insert
            return dict(zip([d[0] for d in cursor.description], result))


    def update_report(self, email, url):
        """Update the report URL for a given email."""
        self.ensure_connection()
        with self.connection.cursor() as cursor:
            cursor.execute("""
                UPDATE reports_fastapi
                SET url = %s, updated_at = NOW()
                WHERE email = %s
                RETURNING id, email, url, updated_at
            """, (url, email))
            result = cursor.fetchone()
            return dict(zip([d[0] for d in cursor.description], result))

    def get_report_by_email(self, email):
        self.ensure_connection()
        with self.connection.cursor() as cursor:
            cursor.execute("""
                SELECT id, email, url, created_at, updated_at
                FROM reports_fastapi
                WHERE email = %s
            """, (email,))
            rows = cursor.fetchall()
            return [dict(zip([d[0] for d in cursor.description], row)) for row in rows]

    def get_reports_by_ids_and_email(self, email, report_ids):
        self.ensure_connection()
        with self.connection.cursor() as cursor:
            query = """
                SELECT id, email, url, created_at, updated_at
                FROM reports_fastapi
                WHERE email = %s
                  AND id = ANY(%s)
            """
            cursor.execute(query, (email, report_ids))
            rows = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]
            return [dict(zip(columns, row)) for row in rows]

    def delete_report_by_email(self, email):
        self.ensure_connection()
        with self.connection.cursor() as cursor:
            cursor.execute("DELETE FROM reports_fastapi WHERE email = %s", (email,))
            return f"{cursor.rowcount} reports_fastapi deleted"

    def delete_user_report_by_id(self, email, report_id):
        self.ensure_connection()
        with self.connection.cursor() as cursor:
            cursor.execute("DELETE FROM reports_fastapi WHERE id = %s AND email = %s", (report_id, email))
            return f"{cursor.rowcount} reports_fastapi deleted"

    def delete_all_tables(self):
        self.ensure_connection()
        with self.connection.cursor() as cursor:
            cursor.execute("DROP TABLE IF EXISTS reports_fastapi")
            cursor.execute("DROP TABLE IF EXISTS akio_data_fastapi")
            print("All tables dropped successfully.")
            return "All tables dropped"


# Sample config - replace with your actual database credentials
PGHOST = 'ep-yellow-recipe-a5fny139.us-east-2.aws.neon.tech'
PGDATABASE = 'test'
PGUSER = 'test_owner'
PGPASSWORD = 'tcWI7unQ6REA'

if __name__ == '__main__':
    pdd = PostgresDatabase()
    pdd.create_connection(PGUSER, PGPASSWORD, PGDATABASE, PGHOST)
    #pdd.delete_all_tables()      # Drops old tables (with all constraints)
    pdd.create_table()           # Creates akio_data_fastapi (no unique on email)
    pdd.create_reports_table()   # Creates reports_fastapi (no unique/FK on email)
    print("Tables recreated with no unique or foreign key constraints on email.")
