import os
import psycopg2
from psycopg2.extras import RealDictCursor

def get_connection():
    conn = psycopg2.connect(
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT", "5432"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        dbname=os.getenv("DB_NAME")
    )
    return conn

def insert_invoice(data: dict) -> int:
    """
    Inserts an invoice into the data_of_invoices table.
    Expects data to be a dictionary with keys: invoice_number, customer, supplier,
    invoice_date, total_amount, items, vector_embedding.
    Returns the inserted record’s ID.
    """
    conn = get_connection()
    cur = conn.cursor()
    query = """
    INSERT INTO data_of_invoices 
    (invoice_number, customer, supplier, invoice_date, total_amount, items, vector_embedding)
    VALUES (%s, %s, %s, %s, %s, %s, %s)
    RETURNING customer_id;
    """
    cur.execute(query, (
        data.get("invoice_number"),
        data.get("customer"),
        data.get("supplier"),
        data.get("invoice_date"),
        data.get("total_amount"),
        data.get("items"),
        data.get("vector_embedding")
    ))
    result = cur.fetchone()
    if result is None:
        conn.rollback()
        cur.close()
        conn.close()
        raise Exception("Insert invoice query did not return any row.")
    inserted_id = result[0]
    conn.commit()
    cur.close()
    conn.close()
    return inserted_id

def insert_invoice_image(invoice_id: int, image_path: str, ocr_text: str) -> int:
    """
    Inserts a record into the data_invoices_images table.
    """
    conn = get_connection()
    cur = conn.cursor()
    query = """
    INSERT INTO data_invoices_images (invoice_id, image_path, ocr_text)
    VALUES (%s, %s, %s)
    RETURNING id;
    """
    cur.execute(query, (invoice_id, image_path, ocr_text))
    result = cur.fetchone()
    if result is None:
        conn.rollback()
        cur.close()
        conn.close()
        raise Exception("Insert invoice image query did not return any row.")
    inserted_id = result[0]
    conn.commit()
    cur.close()
    conn.close()
    return inserted_id

# in db/db_handler.py
def insert_chat_message(invoice_id, role, content, timestamp):
    conn = get_connection()
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO chat_history (invoice_id, role, content, timestamp)
            VALUES (%s, %s, %s, %s)
            """,
            (invoice_id, role, content, timestamp)
        )
        conn.commit()
    conn.close()
