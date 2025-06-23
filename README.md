# Invoice-chatbot
The Retail Invoice Chatbot is an advanced, AI-powered assistant designed to extract, store, and query structured information from invoice images. It combines OCR, natural language understanding, vector embeddings, and a large language model (LLM) to support seamless human-like interactions with retail invoice data.

# 🔍 What It Does

📸 Uploads & Extracts: Accepts invoice images, uses OCR (Tesseract/OpenCV) to extract text.
🧠 Parses with LLM: Sends OCR text to Gemini LLM to extract structured fields like:
Invoice Number, Date, Customer Name, Supplier, Items, Total Amount
🧮 Stores Smartly: Saves parsed invoices and embeddings to a PostgreSQL database and FAISS vector store.

# 💬 Natural Language Chat: Lets users ask flexible questions like:

“What is the total of the latest invoice?”
“Show me the 5th invoice uploaded.”
“Summarize invoice number AMD-2024-559.”
“Who bought from Amazon?”
“What did customer ID 12 purchase?”

# ⚡ Fully LLM-driven: 
No manual parsing or conditional logic; all intelligent behaviors come from prompt-engineered LLMs.

# 🧠 Context-aware Memory: 
Maintains conversational history to support follow-up queries like “What about the one before that?” or “Summarize it.”

🧱 Architecture

                        ┌─────────────────────────────┐
                        │     Streamlit Frontend      │
                        └────────────┬────────────────┘
                                     │
                          ┌──────────▼──────────┐
                          │    app.py (UI)      │
                          └──────────┬──────────┘
                                     │
                ┌────────────────────▼─────────────────────┐
                │        OCR + LLM Parsing Layer            │
                ├───────────────────────────────────────────┤
                │ • OCR using Tesseract + OpenCV            │
                │ • Gemini LLM for field extraction         │
                └────────────────────┬──────────────────────┘
                                     │
                ┌────────────────────▼────────────────────┐
                │      PostgreSQL + FAISS Storage Layer    │
                ├──────────────────────────────────────────┤
                │ • data_of_invoices (with customer_id PK) │
                │ • data_invoices_images (linked via FK)   │
                │ • vector_embedding column (float8[])     │
                └────────────────────┬─────────────────────┘
                                     │
                  ┌──────────────────▼────────────────────┐
                  │     Query Engine (query_llm.py)       │
                  ├───────────────────────────────────────┤
                  │ • Gemini-driven prompt handling        │
                  │ • Positional & ID lookups              │
                  │ • Customer/Supplier/Item logic         │
                  │ • Chat history-aware responses         │
                  └────────────────────────────────────────┘



# 💡 Features Highlights

✅ Works with any invoice number format (INV-, AMD-, numeric, etc.)
✅ Supports deleted/missing IDs by distinguishing between position and primary-key
✅ Accurate for both invoice lookups and contextual questions
✅ Handles time-based queries (e.g., "invoices from last month")
✅ Concise but complete responses with proper formatting
✅ Includes detailed prompt-engineering for reliability

# 🗂 Database Design

# Table: data_of_invoices

Column	Type	Description

customer_id	SERIAL	Primary key,
invoice_number	TEXT	Unique invoice identifier,
customer	TEXT	Customer name,
supplier	TEXT	Supplier name,
invoice_date	DATE	Invoice issue date,
items	TEXT	All items (concatenated string),
total_amount	NUMERIC(12,2)	Invoice total,
vector_embedding	FLOAT8[]	Embedding vector (for FAISS search),
uploaded_at	TIMESTAMP	Time of upload

# Table: data_invoices_images

Column	Type	Description

image_id	SERIAL	Primary key,
customer_id	INT	Foreign key to data_of_invoices,
image_path	TEXT	File path or URI,
uploaded_at	TIMESTAMP	Time of image upload


# 🛠 Tech Stack

Frontend: Streamlit
OCR: Tesseract, OpenCV
LLM: Google Gemini (via google.generativeai)
Vector DB: FAISS + sentence-transformers (MiniLM-L6-v2)
Database: PostgreSQL (via psycopg2)
LangChain: RetrievalQA, ConversationalRetrievalChain, memory, prompt templates

## 📁 Project Structure

-app.py
-db/db_handler.py
-img2text/img2text.py
-llm/query_llm.py

# 🚀 Getting Started

1. Clone this repository
2. Set up PostgreSQL and create the tables (see schema above)
3. Install dependencies: pip install -r requirements.txt
4. Set environment variables:
  GEMINI_API_KEY
  DB_HOST, DB_USER, DB_PASSWORD, DB_NAME
5. Run the chatbot:
  streamlit run app.py

# 📌 Example Questions You Can Ask

What is the latest invoice?
Summarize invoice number AMD2-6095263
Show the 3rd invoice
What did Jacob John buy?
Summarize invoice ID 12
Which customer bought from Flipkart?
Invoices from March 2025?
What’s the average total?
Customer ID 3 details

# 📣 Notes

Handles deletion/missing IDs cleanly (e.g., ID 10 not found).
Distinguishes between invoice ID, invoice number, and upload position.
Customer IDs are now the new primary key and used in image relationships.
