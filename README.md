# Invoice-chatbot
The Retail Invoice Chatbot is an advanced, AI-powered assistant designed to extract, store, and query structured information from invoice images. It combines OCR, natural language understanding, vector embeddings, and a large language model (LLM) to support seamless human-like interactions with retail invoice data.

# 🔍 What It Does

📸 Uploads & Extracts: Accepts invoice images, uses OCR (Tesseract/OpenCV) to extract text.<br>
🧠 Parses with LLM: Sends OCR text to Gemini LLM to extract structured fields like:<br>
Invoice Number, Date, Customer Name, Supplier, Items, Total Amount<br>
🧮 Stores Smartly: Saves parsed invoices and embeddings to a PostgreSQL database and FAISS vector store.<br>

# 💬 Natural Language Chat: Lets users ask flexible questions like:

“What is the total of the latest invoice?”<br>
“Show me the 5th invoice uploaded.”<br>
“Summarize invoice number AMD-2024-559.”<br>
“Who bought from Amazon?”<br>
“What did customer ID 12 purchase?”<br>

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

### Table: data_of_invoices

<b>Column	Type	Description</b>

customer_id	SERIAL	Primary key,<br>
invoice_number	TEXT	Unique invoice identifier,<br>
customer	TEXT	Customer name,<br>
supplier	TEXT	Supplier name,<br>
invoice_date	DATE	Invoice issue date,<br>
items	TEXT	All items (concatenated string),<br>
total_amount	NUMERIC(12,2)	Invoice total,<br>
vector_embedding	FLOAT8[]	Embedding vector (for FAISS search),<br>
uploaded_at	TIMESTAMP	Time of upload<br>

### Table: data_invoices_images

<b>Column	Type	Description</b>

image_id	SERIAL	Primary key,<br>
customer_id	INT	Foreign key to data_of_invoices,<br>
image_path	TEXT	File path or URI,<br>
uploaded_at	TIMESTAMP	Time of image upload<br>


# 🛠 Tech Stack

Frontend: Streamlit<br>
OCR: Tesseract, OpenCV<br>
LLM: Google Gemini (via google.generativeai)<br>
Vector DB: FAISS + sentence-transformers (MiniLM-L6-v2)<br>
Database: PostgreSQL (via psycopg2)<br>
LangChain: RetrievalQA, ConversationalRetrievalChain, memory, prompt templates<br>

## 📁 Project Structure

-app.py<br>
-db/db_handler.py<br>
-img2text/img2text.py<br>
-llm/query_llm.py<br>

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

What is the latest invoice?<br>
Summarize invoice number AMD2-6095263<br>
Show the 3rd invoice<br>
What did Jacob John buy?<br>
Summarize invoice ID 12<br>
Which customer bought from Flipkart?<br>
Invoices from March 2025?<br>
What’s the average total?<br>
Customer ID 3 details<br>

# 📣 Notes

Handles deletion/missing IDs cleanly (e.g., ID 10 not found).<br>
Distinguishes between invoice ID, invoice number, and upload position.<br>
Customer IDs are now the new primary key and used in image relationships.<br>
