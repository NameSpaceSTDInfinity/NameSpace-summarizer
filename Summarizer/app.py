# app.py (Flask backend)

from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
from transformers import T5Tokenizer, T5ForConditionalGeneration
from pdfminer.high_level import extract_text

app = Flask(__name__)
UPLOAD_FOLDER = './uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load pre-trained T5 model and tokenizer
model = T5ForConditionalGeneration.from_pretrained('./fine_tuned_t5_model')
tokenizer = T5Tokenizer.from_pretrained('./fine_tuned_t5_model')

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    text = extract_text(pdf_path)
    return text

# Function to summarize text
def summarize_text(text, max_length=150):
    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=512, truncation=True)
    summary_ids = model.generate(inputs, max_length=max_length, min_length=30, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Save the file
    filename = secure_filename(file.filename)
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(file_path)

    # Extract text and summarize
    extracted_text = extract_text_from_pdf(file_path)
    summary = summarize_text(extracted_text)

    return jsonify({"summary": summary})

if __name__ == '__main__':
    app.run(debug=True)
