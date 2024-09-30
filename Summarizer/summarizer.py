import os
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, AdamW
from pdfminer.high_level import extract_text

# Define model directory path to save/load the fine-tuned model
fine_tuned_model_dir = "./fine_tuned_t5_model"
model_name = "t5-small"

# Load the T5 tokenizer
tokenizer = T5Tokenizer.from_pretrained(model_name)

# Check if CUDA is available for faster computation, else use CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Function to fine-tune the model (if it hasn’t been fine-tuned yet)
def fine_tune_model(model, dataset, epochs=1, learning_rate=1e-4):
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    model.train()

    for epoch in range(epochs):
        for text in dataset:
            # Tokenize and prepare input
            inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True, padding=True).to(model.device)
            labels = inputs.input_ids

            # Forward pass and loss calculation
            outputs = model(input_ids=inputs.input_ids, labels=labels)
            loss = outputs.loss

            # Backward pass and optimizer step
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    # Save the fine-tuned model
    model.save_pretrained(fine_tuned_model_dir)
    tokenizer.save_pretrained(fine_tuned_model_dir)
    print("Model fine-tuned and saved!")

# Function to load the pre-trained or fine-tuned model
def load_model():
    if os.path.exists(fine_tuned_model_dir):
        print("Loading fine-tuned model...")
        model = T5ForConditionalGeneration.from_pretrained(fine_tuned_model_dir)
    else:
        print("No fine-tuned model found. Fine-tuning model now...")
        model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)

        # Dummy dataset to simulate fine-tuning (you can replace this with real data)
        dataset = ["Sample text for fine-tuning", "Another piece of text for tuning"]
        fine_tune_model(model, dataset)
    return model.to(device)

# Function to extract text from PDF using pdfminer
def extract_text_from_pdf(pdf_path):
    text = extract_text(pdf_path)  # Extract text directly
    return text

# Function to summarize text using T5
def summarize_text(text, model, max_length=150):
    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=512, truncation=True).to(device)
    summary_ids = model.generate(inputs, max_length=max_length, min_length=30, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Function to save summary as a text file
def save_summary_to_txt(txt_filename, summary):
    with open(txt_filename, 'w') as f:
        f.write(summary)
    print(f"Summary saved as {txt_filename}")

# Main function to upload PDF, extract, and summarize
def main():
    # Load the model (fine-tuned if exists, or pre-trained)
    model = load_model()

    # Upload a PDF file path manually (or automate this in a file upload interface)
    pdf_path = "Summarizer/Assignment1.pdf"

    # Extract text from the uploaded PDF
    extracted_text = extract_text_from_pdf(pdf_path)
    print(f"Extracted Text:\n{extracted_text[:2000]}")  # Showing first 2000 characters

    # Summarize the extracted text
    summary = summarize_text(extracted_text, model)
    print(f"\n\nSummary:\n{summary}")

    # Ask the user if they want to save the summary as a text file
    save_as_txt = input("Do you want to save the summary as a text file? (yes/no): ").strip().lower()
    if save_as_txt == 'yes':
        txt_filename = f"{os.path.splitext(pdf_path)[0]}_summary.txt"
        save_summary_to_txt(txt_filename, summary)

if __name__ == "__main__":
    main()
