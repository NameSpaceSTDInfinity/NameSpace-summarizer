from flask import Flask, request, render_template
from transformers import T5Tokenizer, T5ForConditionalGeneration
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW
import torch
import PyPDF2

app = Flask(__name__)

# Load pre-trained T5 model and tokenizer
model_name = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)
model = model.to("cuda" if torch.cuda.is_available() else "cpu")

# Function to extract text from PDF using PyPDF2
def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Custom Dataset class to handle the text and summary
class CustomDataset(Dataset):
    def __init__(self, texts, summaries, tokenizer, max_length=512):
        self.texts = texts
        self.summaries = summaries
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        source = self.tokenizer(self.texts[index], truncation=True, padding='max_length', max_length=self.max_length, return_tensors="pt")
        target = self.tokenizer(self.summaries[index], truncation=True, padding='max_length', max_length=self.max_length, return_tensors="pt")
        return {
            'input_ids': source.input_ids.flatten(),
            'attention_mask': source.attention_mask.flatten(),
            'labels': target.input_ids.flatten()
        }

# Fine-tune the T5 model on the extracted text
def fine_tune_t5_on_pdf(texts, summaries, model, tokenizer, epochs=1, batch_size=8):
    dataset = CustomDataset(texts, summaries, tokenizer)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = AdamW(model.parameters(), lr=5e-5)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.train()
    for epoch in range(epochs):
        for batch in loader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

    return model

# Generate summary using fine-tuned T5 model
def generate_summary(text, model, tokenizer, max_length=150):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=512, truncation=True).to(device)
    summary_ids = model.generate(inputs, max_length=max_length, min_length=30, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Flask routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def summarize():
    pdf_file = request.files['pdf']
    extracted_text = extract_text_from_pdf(pdf_file)

    # Placeholder summaries for initial fine-tuning
    summaries = ["This is a placeholder summary for fine-tuning."] * len(extracted_text.split(".")[:5])  # Placeholder summary

    # Fine-tune the model on the extracted text
    fine_tune_t5_on_pdf([extracted_text], summaries, model, tokenizer)

    # Generate summary from the fine-tuned model
    summary = generate_summary(extracted_text, model, tokenizer)

    return render_template('index.html', summary=summary)

if __name__ == '__main__':
    app.run(debug=True)
