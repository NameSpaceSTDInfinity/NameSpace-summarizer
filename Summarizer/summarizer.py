import streamlit as st
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from PyPDF2 import PdfReader
import json
from fpdf import FPDF

# Load T5 model and tokenizer
model_name = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)
model = model.to("cuda" if torch.cuda.is_available() else "cpu")

# CSS Styling
st.markdown("""
    <style>
        /* Custom style for title */
        .title {
            font-size: 2.5em;
            font-weight: bold;
            text-align: center;
            color: #4A90E2;
            margin-top: 0.5em;
            margin-bottom: 0.5em;
        }
        
        /* File uploader and text area styling */
        .stFileUploader, .stTextArea {
            background-color: #f7f9fc;
            border: 1px solid #cfcfcf;
            padding: 10px;
            border-radius: 5px;
        }

        /* Button styling */
        .stButton > button {
            background-color: #4A90E2;
            color: #FFFFFF;
            border: none;
            padding: 0.5em 1em;
            font-size: 1em;
            border-radius: 10px;
            transition: 0.3s;
        }
        
        .stButton > button:hover {
            background-color: #357ABD;
        }
        
        /* Download button styling */
        .download-button {
            background-color: #ff6b6b;
            color: white;
            padding: 10px;
            border-radius: 10px;
            font-size: 1em;
            text-decoration: none;
            display: inline-block;
            margin-top: 10px;
        }
    </style>
""", unsafe_allow_html=True)

# Extract text from PDF
def extract_text_from_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# Extract text from JSON
def extract_text_from_json(file):
    data = json.load(file)
    return json.dumps(data, indent=2)  # Converts JSON data to a readable string

# Summarize text using T5 model
def summarize_text(text, max_length=150):
    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=512, truncation=True).to(model.device)
    summary_ids = model.generate(inputs, max_length=max_length, min_length=30, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Save summary to PDF
def save_summary_as_pdf(summary_text, filename="summary.pdf"):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, summary_text)
    pdf.output(filename)

# Streamlit interface
st.markdown('<h1 class="title">📄 Text Summarization App</h1>', unsafe_allow_html=True)
st.write("Upload a PDF or JSON file, and this app will extract the text and generate a summary!")

uploaded_file = st.file_uploader("Choose a PDF or JSON file", type=["pdf", "json"])

if uploaded_file is not None:
    # Determine file type and extract text
    file_type = uploaded_file.name.split(".")[-1]
    if file_type == "pdf":
        extracted_text = extract_text_from_pdf(uploaded_file)
    elif file_type == "json":
        extracted_text = extract_text_from_json(uploaded_file)
    else:
        st.error("Unsupported file type.")
        st.stop()
    
    # Display extracted text
    st.subheader("Extracted Text")
    st.text_area("Text from file", extracted_text, height=300)
    
    # Generate summary
    if st.button("Generate Summary"):
        with st.spinner("Summarizing..."):
            summary = summarize_text(extracted_text)
        st.subheader("Summary")
        st.write(summary)
        
        # Save summary as PDF and download
        if st.button("Save Summary as PDF"):
            save_summary_as_pdf(summary)
            with open("summary.pdf", "rb") as pdf_file:
                st.download_button(label="📥 Download Summary as PDF", data=pdf_file, file_name="summary.pdf", mime="application/pdf")
