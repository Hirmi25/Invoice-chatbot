import os
from datetime import datetime
import json
import streamlit as st
from langchain.schema import HumanMessage, AIMessage
from llm.query_llm import run_query, run_chat, llm_extract_invoice_fields
from img2text.img2text import extract_text_from_image
from db.db_handler import insert_invoice, insert_invoice_image, insert_chat_message
from langchain.embeddings import HuggingFaceEmbeddings

# Configure Streamlit.
st.set_page_config(page_title="Robust Invoice Chatbot", layout="wide")
st.title("Robust Invoice Chatbot")

# Initialize session state.
if "last_uploaded_name" not in st.session_state:
    st.session_state.last_uploaded_name = ""
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "current_invoice_id" not in st.session_state:
    st.session_state.current_invoice_id = None

# --------------------------------------------------
# Invoice Upload Section (Sidebar)
# --------------------------------------------------
st.sidebar.header("Upload Invoice")
st.sidebar.write("Upload an invoice image for extraction.")

uploaded_file = st.sidebar.file_uploader("Choose an invoice image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None and uploaded_file.name != st.session_state.last_uploaded_name:
    st.session_state.last_uploaded_name = uploaded_file.name
    st.session_state.chat_history = []  # Clear previous conversation.

    # Save the uploaded file.
    temp_folder = "temp_upload"
    os.makedirs(temp_folder, exist_ok=True)
    file_path = os.path.join(temp_folder, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.sidebar.success("Invoice uploaded and stored successfully!")

    # Extract OCR text.
    ocr_text = extract_text_from_image(file_path)

    # LLM-based extraction.
    extracted = llm_extract_invoice_fields(ocr_text)

    # Post-process extracted data: Force missing keys to "N/A" or "0"
    required_fields = ["invoice_number", "invoice_date", "total_amount", "supplier", "customer", "items"]
    for key in required_fields:
        if key not in extracted or not extracted.get(key) or extracted.get(key).strip() == "":
            extracted[key] = "0" if key == "total_amount" else "N/A"

    # Format invoice_date into YYYY-MM-DD
    if extracted.get("invoice_date") and extracted["invoice_date"] != "N/A":
        parsed_date = None
        for fmt in ("%d/%m/%Y", "%d-%m-%Y"):  # support 06/05/2025 or 06-05-2025
            try:
                parsed_date = datetime.strptime(extracted["invoice_date"], fmt)
                break
            except Exception:
                continue
        extracted["invoice_date"] = parsed_date.strftime("%Y-%m-%d") if parsed_date else datetime.today().strftime("%Y-%m-%d")
    else:
        extracted["invoice_date"] = datetime.today().strftime("%Y-%m-%d")

    # Compute vector embedding
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    invoice_content = (
        f"Invoice Number: {extracted.get('invoice_number')}\n"
        f"Customer: {extracted.get('customer')}\n"
        f"Supplier: {extracted.get('supplier')}\n"
        f"Date: {extracted.get('invoice_date')}\n"
        f"Items: {extracted.get('items')}\n"
        f"Total Amount: {extracted.get('total_amount')}"
    )
    vector = embeddings.embed_query(invoice_content)
    extracted["vector_embedding"] = json.dumps(vector)

    # Insert invoice into DB
    inserted_id = insert_invoice(extracted)
    if inserted_id:
        st.session_state.current_invoice_id = inserted_id
        try:
            insert_invoice_image(inserted_id, file_path, ocr_text)
        except Exception as e:
            print("Error inserting invoice image:", e)
    else:
        st.sidebar.error("Failed to save invoice. Check the console for details.")

# --------------------------------------------------
# Chat Interface Section
# --------------------------------------------------
st.header("Chat with Your Invoices")

chat_placeholder = st.empty()


def render_chat():
    """Render the entire chat conversation."""
    chat_content = ""
    for msg in st.session_state.chat_history:
        if msg.__class__.__name__ == "HumanMessage":
            chat_content += f"**You:** {msg.content}\n\n"
        elif msg.__class__.__name__ == "AIMessage":
            chat_content += f"**Bot:** {msg.content}\n\n"
    chat_placeholder.markdown(chat_content)


render_chat()

with st.form(key="chat_form", clear_on_submit=True):
    query = st.text_input("Your message:")
    submit_button = st.form_submit_button("Send")

if submit_button:
    if query.strip():
        with st.spinner("Processing your query..."):
            answer, updated_history = run_chat(query, st.session_state.chat_history)
            # Update session state with new history
            st.session_state.chat_history = updated_history
            # Persist chat history in DB for later context
            if st.session_state.current_invoice_id:
                insert_chat_message(
                    st.session_state.current_invoice_id,
                    'user',
                    query,
                    datetime.utcnow()
                )
                insert_chat_message(
                    st.session_state.current_invoice_id,
                    'bot',
                    answer,
                    datetime.utcnow()
                )
        render_chat()
    else:
        st.warning("Please enter a valid query.")
