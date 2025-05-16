from flask import Flask, render_template, request, send_file
import pickle
import numpy as np
import pandas as pd
import os
import csv
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from preprocessing.preprocessing import cleaningText, casefoldingText, fix_slangwords, tokenizingText, filteringText, stemmingText




app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)


model = load_model('model/best_lstm_modelV2.h5')

with open('model/tokenizer_lstmV2.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

with open('model/label_encoder_lstmV2.pickle', 'rb') as handle:
    label_encoder = pickle.load(handle)

@app.route('/')
def index():
    return render_template('index.html')

def preprocess_new_text(text):
    cleaned = cleaningText(text)
    casefolded = casefoldingText(cleaned)
    fixed = fix_slangwords(casefolded)
    tokenized = tokenizingText(fixed)
    filtered = filteringText(tokenized)
    stemmed = stemmingText(filtered)
    return stemmed

@app.route('/analyze_csv', methods=['POST'])
def analyze_csv():
    if 'file' not in request.files:
        return "No file part"
    
    file = request.files['file']
    if file.filename == '':
        return "No selected file"
    if not (file.filename.endswith('.xlsx') or file.filename.endswith('.xls')):
        return "Only Excel files (.xlsx or .xls) are supported"
    
    filename= secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    try:
        df = pd.read_excel(filepath)
    except Exception as e:
        return f"Gagal membaca file Excel: {e}"

    if 'content' not in df.columns:
        return "Excel file must contain a 'content' column"
    
    df['stemmed'] = df['content'].astype(str).apply(preprocess_new_text)

    sequences = tokenizer.texts_to_sequences(df['stemmed'].tolist())
    padded = pad_sequences(sequences, maxlen=200)


    predictions = model.predict(padded)
    predictions_classes = np.argmax(predictions, axis=1)
    df['sentiment'] = label_encoder.inverse_transform(predictions_classes)

    result_filename = f'result_{filename}'
    result_path = os.path.join(RESULT_FOLDER, result_filename)
    df.to_excel(result_path, index=False)

    sentiment_counts = df['sentiment'].value_counts(normalize=True)*100
    sentiment_percentage = {
    'positif': round(sentiment_counts.get('positif', 0), 2),
    'negatif': round(sentiment_counts.get('negatif', 0), 2),
    'netral': round(sentiment_counts.get('netral', 0), 2)
    }

    return render_template(
        'outputPage.html',
        percentages=sentiment_percentage,
        download_link=result_filename,
        )

@app.route('/download/<filename>')
def download_result(filename):
    filepath = os.path.join(RESULT_FOLDER, filename)
    return send_file(filepath, as_attachment=True)

@app.route('/analyze_text', methods=['POST'])
def analyze_text():
    text = request.form['text']

    raw = text
    cleaned = cleaningText(raw)
    casefolded = casefoldingText(cleaned)
    fixed_slang = fix_slangwords(casefolded)
    tokenized = tokenizingText(fixed_slang)
    filtered = filteringText(tokenized)
    stemmed = stemmingText(filtered)

    sequence = tokenizer.texts_to_sequences([stemmed])
    padded_sequence = pad_sequences(sequence, maxlen=200)
    
    # Prediksi menggunakan model
    prediction = model.predict(padded_sequence)
    prediction_label = np.argmax(prediction, axis=1)
    decoded_label = label_encoder.inverse_transform(prediction_label)

    return render_template('outputPage.html', sentiment=decoded_label[0], text=text, stemmed=stemmed)

if __name__ == '__main__':
    app.run(debug=True)
