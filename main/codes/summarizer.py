# Import libraries
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from PyPDF2 import PdfReader
# from google.colab import files
import os

# Load the T5 model and tokenizer
model_name = "t5-large"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)
model = model.to("cuda" if torch.cuda.is_available() else "cpu")

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# Function to summarize text using T5
def summarize_text(text, max_length=150):
    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=512, truncation=True).to(model.device)
    summary_ids = model.generate(inputs, max_length=max_length, min_length=30, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Upload PDF file
uploaded = files.upload()

# Save uploaded PDF
pdf_filename = next(iter(uploaded))
pdf_path = pdf_filename

# Extract text from uploaded PDF
extracted_text = extract_text_from_pdf(pdf_path)
print(f"Extracted Text:\n{extracted_text[:2000]}")  # Showing first 2000 characters of extracted text

# Summarize the extracted text
summary = summarize_text(extracted_text)
print(f"\n\nSummary:\n{summary}")
