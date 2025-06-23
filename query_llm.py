import os
import re
import json
import faiss
import google.generativeai as genai
from dotenv import load_dotenv
from langchain.docstore.document import Document
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.llms.base import LLM
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from db.db_handler import get_connection
import datetime
from datetime import datetime, timezone  # Add timezone awareness
from langchain_core.prompts import PromptTemplate
        
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

class GeminiLLM(LLM):
    @property
    def _llm_type(self) -> str:
        return "gemini"
    
    def _call(self, prompt: str, stop=None) -> str:
        # Prepend the system instruction for today's date
        date_instruction = "Today's date is 06 May 2025. Use this as the current date in all time-based reasoning.\n\n"
        full_prompt = date_instruction + prompt

        # Adjust the prompt to interpret questions like "first entry" or "last entry"
        if "first entry" in prompt.lower() or "first invoice" in prompt.lower():
            full_prompt += "\nYou should consider the earliest uploaded invoice (entry 1 as the first)."
        elif "last entry" in prompt.lower() or "latest invoice" in prompt.lower():
            full_prompt += "\nYou should consider the most recent uploaded invoice (last entry as the latest)."
        
        # Call Gemini API to generate the response
        response = genai.GenerativeModel("gemini-2.0-flash").generate_content(full_prompt)
        return response.text.strip()
    
    @property
    def device(self) -> str:
        return "cpu"



def llm_extract_invoice_fields(ocr_text: str) -> dict:
    """
    Use the Gemini LLM exclusively to extract invoice fields.
    The prompt instructs the model to return a valid JSON object with
    the following keys: invoice_number, invoice_date, total_amount, supplier, customer, items.
    If a field cannot be extracted, output "N/A" for that field.
    
    This function preprocesses the response by extracting the part starting
    at the first "{" so that any extra text (such as a leading "json") is removed.
    """
    prompt = f"""
You are an expert invoice extraction assistant. Given the OCR text below, extract the following details and ensure each field has a valid value (if not available, output "N/A"):
IMPORTANT: Today's date is 06-05-2025 (6th May 2025). DO NOT use the current system date or your internal date. Always assume 06-05-2025 is today.
- invoice_number
- invoice_date (in DD-MM-YYYY or DD/MM/YYYY format)
- total_amount
- supplier
- customer
- items (concatenate item descriptions separated by " | ")

Return only a valid JSON object with exactly these keys in this format:
{{"invoice_number": "<value>", "invoice_date": "<value>", "total_amount": "<value>", "supplier": "<value>", "customer": "<value>", "items": "<value>"}}
Do not output any additional commentary.

OCR Text:
{ocr_text}
"""
    llm_model = GeminiLLM()
    response = llm_model._call(prompt).strip()
    if not response:
        print("LLM returned an empty response.")
        return {}
    # Remove possible markdown code block wrappers.
    if response.startswith("```") and response.endswith("```"):
        response = response.strip("`").strip()
    # If there is any extra text before the JSON, find the first '{'
    json_start = response.find("{")
    if json_start != -1:
        response = response[json_start:]
    try:
        data = json.loads(response)
    except Exception as e:
        print("LLM extraction error:", e)
        print("Response was:", response)
        data = {}
    return data

# --- Database Retrieval and Chat Functions ---
def fetch_all_invoices():
    invoices = []
    try:
        conn = get_connection()
        with conn.cursor() as cur:
            query = """
            SELECT customer_id, invoice_number, customer, supplier,
                to_char(invoice_date, 'YYYY-MM-DD') as invoice_date,
                items, total_amount, vector_embedding
            FROM data_of_invoices
            ORDER BY customer_id ASC;  -- Sort by customer_id
            """
            cur.execute(query)
            rows = cur.fetchall()
            for row in rows:
                invoice = {
                    "customer_id": row[0],  # Added as first column
                    "invoice_number": row[1],  # Shifted from row[0] → row[1]
                    "customer": row[2],       # Shifted from row[1] → row[2]
                    "supplier": row[3],       # Shifted from row[2] → row[3]
                    "invoice_date": row[4],   # Shifted from row[3] → row[4]
                    "items": row[5],          # Shifted from row[4] → row[5]
                    "total_amount": row[6],   # Shifted from row[5] → row[6]
                    "vector_embedding": row[7] # Shifted from row[6] → row[7]
                }
                invoices.append(invoice)
        conn.close()
    except Exception as e:
        print("Error fetching all invoices:", e)
    return invoices

def get_all_invoice_documents():
    invoices = fetch_all_invoices()
    docs = []
    for idx, inv in enumerate(invoices, 1):  # Remove reversed() to keep DB order
        content = (
            f"Customer ID: {inv['customer_id']}\n" 
            f"Upload Order: {idx} (1 = Oldest, {len(invoices)} = Newest)\n"
            f"Invoice Number: {inv['invoice_number']}\n"
            f"Customer: {inv['customer']}\n"
            f"Supplier: {inv['supplier']}\n"
            f"Date: {inv['invoice_date']}\n"
            f"Items: {inv['items']}\n"
            f"Total Amount: {inv['total_amount']}"
        )
        docs.append(Document(page_content=content))
    return docs

def fetch_latest_invoice():
    try:
        conn = get_connection()
        with conn.cursor() as cur:
            query = """
            SELECT customer_id, invoice_number, customer, supplier,
                to_char(invoice_date, 'YYYY-MM-DD') as invoice_date,
                items, total_amount, vector_embedding
            FROM data_of_invoices
            ORDER BY customer_id DESC
            LIMIT 1;
            """
            cur.execute(query)
            row = cur.fetchone()
            if row:
                invoice = {
                    "customer_id": row[0],  # Added as first column
                    "invoice_number": row[1],  # Shifted from row[0] → row[1]
                    "customer": row[2],       # Shifted from row[1] → row[2]
                    "supplier": row[3],       # Shifted from row[2] → row[3]
                    "invoice_date": row[4],   # Shifted from row[3] → row[4]
                    "items": row[5],          # Shifted from row[4] → row[5]
                    "total_amount": row[6],   # Shifted from row[5] → row[6]
                    "vector_embedding": row[7] # Shifted from row[6] → row[7]
                }
                conn.close()
                return invoice
        conn.close()
    except Exception as e:
        print("Error fetching latest invoice:", e)
    return None

def get_latest_invoice_document():
    invoice = fetch_latest_invoice()
    if invoice:
        content = (
            f"Invoice Number: {invoice['invoice_number']}\n"
            f"Customer: {invoice['customer']}\n"
            f"Supplier: {invoice['supplier']}\n"
            f"Date: {invoice['invoice_date']}\n"
            f"Items: {invoice['items']}\n"
            f"Total Amount: {invoice['total_amount']}\n"
            f"Customer ID: {invoice['customer_id']}"
        )
        return [Document(page_content=content)]
    return []



def create_vectorstore_from_docs(docs):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore


def run_query(query: str) -> str:
    # Load all invoice documents
    docs = get_all_invoice_documents()
    total_docs = len(docs)

    # Vector similarity search
    vectorstore = create_vectorstore_from_docs(docs)
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": total_docs}
    )

    # Core prompt -- VERY explicit rules
    qa_prompt = f"""You are InvoiceBot. Follow these rules for **every** question:

-0. **Pronoun Resolution**  
-   - If the user says “summarize it,” “summarize this invoice,” or uses “it”/“this”/“that” referring to an invoice, assume they mean the **most recent** invoice.
+    memory.chat_memory.add_message(
+        SystemMessage(
+            content=You are InvoiceBot. Follow these rules for **every** question:
+
+0. **Explicit-Invoice Follow-Up**  
+   - If the user’s previous query explicitly named an invoice (by invoice number or customer name), then any unqualified follow-up (“what is invoice number?”, “what date?”, etc.) refers to **that** invoice.
+
+1. **Pronoun Resolution**  
+   - Only if the user uses a pronoun (“it”, “this”, “that”) or asks “summarize it/this invoice”, assume they mean the **most recent** uploaded invoice.
 
1. **ID vs. Position**  
   - If the user specifies "customer id X", "id X", or "uid X", fetch the invoice where `customer_id = X`.  
   - For "invoice N" (plain integer), use upload position (1 = oldest; LAST = newest).  
   - "first"/"last" always refer to upload order, NOT customer_id.


2. **Customer⇄Supplier Intents**  
   - “Who bought from <Supplier>?” → list customer(s) for that supplier.  
   - “From where did <Customer> buy?” → list supplier(s) for that customer.

3. **Date Filtering & Relative Time**  
   - Support absolute dates (YYYY-MM-DD), ranges (“from A to B”), and relative  
     (“last week/month”, “past 30 days”, “2 months ago”).  

4. **Amount & Items**  
   - Highest/lowest/above/below thresholds.  
   - Count or list items on specific invoices.  
   - Contain-item queries (“contains ‘Headphones’”).

5. **Aggregation & Comparison**  
   - Sum, average, difference, total of N invoices, and “who spent the most.”

6. **Complex Filters**  
   - Odd/even positions, closest to a date, multi-criteria (customer + date range).

7. **Answer Style**  
   - Concise, direct. No apologies or “context doesn’t include.”  
   - Always include minimally: Invoice Number, Customer, Date when detailing invoices.  
   - For supplier/customer queries, single sentence:  
     “<Customer> bought from <Supplier>.”  
   - For aggregations, state the result directly.
   - Do **not** repeat back or confirm user inputs. If they mention “Neha Joshi,” don’t say “Yes, Neha Joshi is a customer name.”  
   - Simply answer the question directly.

8. **Follow-Up Context**  
   - If the user’s previous exchange named or summarized a particular invoice (by customer or invoice number), then any question without a qualifier (“what is invoice number?”) refers to that same invoice.

9. **Metadata Suppression**  
   - Only mention these six fields in your answers:  
     **Invoice Number**, **Customer**, **Supplier**, **Date**, **Items**, **Total Amount**.  
   - Do **not** output internal metadata (Customer ID, Upload Order, vector embeddings, etc.) unless the user explicitly asks for it.

10. **Examples**  
   - Q: “first invoice summary?”  
     → “The oldest invoice is INV-2001 (K. Patel) on 2024-04-16. Total ₹5 658.10; Items: Shoes, Kurta, Headphones, Burger.”  
   - Q: “invoice INV-2003 date?”  
     → “Invoice INV-2003 was issued on 2024-04-27.”  
   - Q: “who bought from Amazon?”  
     → “N. Singh bought from Amazon.”  
   - Q: “last week’s invoices?”  
     → “Invoices from 2025-05-02 to 2025-05-08: INV-2006, INV-2007 (Totals ₹412.00, ₹440.00).”  
   - Q: “difference between invoice 2 and 4?”  
     → “Invoice 2 (₹1 884.46) vs. Invoice 4 (₹2 414.28): difference ₹529.82.”
     - Q: "summarize customer id 2"  
  → "Customer ID 2: Invoice INV-3004 (Ravi Kumar) bought from Amazon India on 2025-01-28. Total ₹590.00; Items: Burger."

Context:  
{{context}}

Question: {{question}}  
Answer:"""

    llm = GeminiLLM()
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": qa_prompt}
    )

    return qa_chain.run(query)



def run_chat(query: str, chat_history: list) -> (str, list):
    # Load all documents
    docs = get_all_invoice_documents()
    total_docs = len(docs)

    # Build retriever
    vectorstore = create_vectorstore_from_docs(docs)
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": total_docs}
    )

    llm = GeminiLLM()
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    memory.chat_memory.messages += chat_history

    # System-level instructions for chat
    memory.chat_memory.add_message(
        SystemMessage(
            content=f"""You are InvoiceBot. Follow these rules for **every** question:

-0. **Pronoun Resolution**  
-   - If the user says “summarize it,” “summarize this invoice,” or uses “it”/“this”/“that” referring to an invoice, assume they mean the **most recent** invoice.
+    memory.chat_memory.add_message(
+        SystemMessage(
+            content=You are InvoiceBot. Follow these rules for **every** question:
+
+0. **Explicit-Invoice Follow-Up**  
+   - If the user’s previous query explicitly named an invoice (by invoice number or customer name), then any unqualified follow-up (“what is invoice number?”, “what date?”, etc.) refers to **that** invoice.
+
+1. **Pronoun Resolution**  
+   - Only if the user uses a pronoun (“it”, “this”, “that”) or asks “summarize it/this invoice”, assume they mean the **most recent** uploaded invoice.
 
1. **ID vs. Position**  
   - "customer id X" → exact `customer_id` lookup.  
   - "invoice N" → upload position N.  
   - Never mix customer_id with upload order.

2. **Customer⇄Supplier Intents**  
   - “Who bought from <Supplier>?” → customer(s) for that supplier.  
   - “From where did <Customer> buy?” → supplier(s) for that customer.

3. **Date & Relative Time**  
   - Handle YYYY-MM-DD, ranges (A to B), and relative (“last week”, “30 days ago”, etc.).

4. **Amount & Items**  
   - Threshold, highest/lowest, item counts, contain-item queries.

5. **Aggregation & Comparison**  
   - Sum, average, difference, top spender.

6. **Complex Filters**  
   - Odd/even positions, closest to date, multi-criteria.

7. **Answer Style**  
   - Concise, direct; no apologies or context-missing statements.  
   - Always include Invoice Number, Customer, Date for invoice details.  
   - Do **not** repeat back or confirm user inputs. If they mention “Neha Joshi,” don’t say “Yes, Neha Joshi is a customer name.”  
    - Simply answer the question directly.
   - Supplier/customer queries in one sentence:  
     “<Customer> bought from <Supplier>.”  
   - Aggregations: state result directly.
8. **Follow-Up Context**  
+   - If the user’s previous exchange named or summarized a particular invoice (by customer or invoice number), then any question without a qualifier (“what is invoice number?”) refers to that same invoice.
+
9. **Metadata Suppression**  
+   - Only mention these six fields in your answers:  
+     **Invoice Number**, **Customer**, **Supplier**, **Date**, **Items**, **Total Amount**.  
+   - Do **not** output internal metadata (Customer ID, Upload Order, vector embeddings, etc.) unless the user explicitly asks for it.
+

10. **Examples**  
   - See `run_query` examples above.

Now answer the user’s question based on chat history and document context."""
        )


    )

    # Rephrase ambiguous queries to be clearer for the LLM
    condense_prompt = PromptTemplate(
        input_variables=["chat_history", "question"],
        template=f"""Rephrase the question to decide:
- Whether it's a direct invoice_number lookup or a position lookup.
- Map "first"/"last"/numeric accordingly.
- Keep invoice IDs (INV-*) intact.
Chat History:
{{chat_history}}
Question: {{question}}
Rephrased Question:"""
    )

    chat_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        condense_question_prompt=condense_prompt,
        condense_question_llm=llm
    )

    result = chat_chain({"question": query})
    answer = result.get("answer", "").strip()

    # Strip out the system messages before returning
    updated_history = [
        msg for msg in memory.chat_memory.messages
        if not isinstance(msg, SystemMessage)
    ]

    return answer, updated_history
