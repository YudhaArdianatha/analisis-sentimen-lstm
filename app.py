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
from google_play_scraper import reviews, Sort
from wordcloud import WordCloud
import uuid
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
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
        return render_template('index.html', error_message="Tidak ada file yang diunggah.")
    
    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', error_message="Nama file kosong.")
    
    if not (file.filename.endswith('.xlsx') or file.filename.endswith('.xls')):
        return render_template('index.html', error_message="Hanya file Excel (.xlsx atau .xls) yang didukung.")
    
    filename= secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    try:
        df = pd.read_excel(filepath)

        # Validasi jika file Excel kosong
        if df.empty:
            return render_template('index.html', error_message="File Excel kosong. Mohon upload file yang berisi data.")
        
        # Validasi jika kolom 'content' tidak ada
        if 'content' not in df.columns:
            return render_template('index.html', error_message="File Excel tidak memiliki kolom 'content'. Pastikan ada kolom bernama 'content' di file Excel Anda.")
        
        # Validasi jika kolom 'content' kosong
        if df['content'].isna().all() or df['content'].str.strip().eq('').all():
            return render_template('index.html', error_message="Kolom 'content' kosong. Pastikan kolom 'content' berisi data ulasan.")
        
    except Exception as e:
        return render_template('index.html', error_message=f"Gagal membaca file Excel: {e}")

    
    df['stemmed'] = df['content'].astype(str).apply(preprocess_new_text)

    jumlah_data = len(df)

    sequences = tokenizer.texts_to_sequences(df['stemmed'].tolist())
    padded = pad_sequences(sequences, maxlen=200)


    predictions = model.predict(padded)
    predictions_classes = np.argmax(predictions, axis=1)
    df['sentiment'] = label_encoder.inverse_transform(predictions_classes)

    confindence_scores = np.max(predictions, axis=1)
    df['confidence_score'] = confindence_scores
    average_confidence = round(float(np.mean(confindence_scores)), 4)

    result_filename = f'result_{filename}'
    result_path = os.path.join(RESULT_FOLDER, result_filename)
    df.to_excel(result_path, index=False)

    sentiment_counts = df['sentiment'].value_counts(normalize=True)*100
    sentiment_percentage = {
    'positif': round(sentiment_counts.get('positif', 0), 2),
    'negatif': round(sentiment_counts.get('negatif', 0), 2),
    'netral': round(sentiment_counts.get('netral', 0), 2)
    }

    wordclod_filename = f'wordcloud_{result_filename}_{uuid.uuid4().hex}.png'
    all_text = ' '.join(df['stemmed'])
    wordcloud_path = os.path.join('static', wordclod_filename)
    generate_wordcloud(all_text, wordcloud_path)

    return render_template(
        'outputPage.html',
        percentages=sentiment_percentage,
        download_link=result_filename,
        jumlah_data=jumlah_data,
        wordcloud_path=f'static/{wordclod_filename}',
        average_confidence=average_confidence,
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
    confidence_score = float(np.max(prediction))

    return render_template('outputPage.html', sentiment=decoded_label[0], text=text, stemmed=stemmed, confidence_score=confidence_score)

@app.route('/analyze_appcode', methods=['POST'])
def analyze_appcode():
    app_code = request.form.get('app_code', '').strip()
    if not app_code:
        return render_template('index.html', error_message="Kode aplikasi tidak boleh kosong.")
    
    try:
        result, _ = reviews(
            app_code,
            lang='id',
            country='ID',
            sort=Sort.NEWEST,
            count=1000
        )
        if not result:
            return render_template('index.html', error_message="Tidak ada ulasan yang ditemukan untuk kode aplikasi ini.")
        
    except Exception as e:
        return render_template('index.html', error_message=f"Gagal mengambil data dari PlayStore: Kode aplikasi '{app_code}' tidak ditemukan atau terjadi kesalahan saat mengakses data.")
    
    
    df = pd.DataFrame(result)
    df = df[['content']]

    df['stemmed'] = df['content'].astype(str).apply(preprocess_new_text)

    jumlah_data = len(df) 

    sequences = tokenizer.texts_to_sequences(df['stemmed'].tolist())
    padded = pad_sequences(sequences, maxlen=200)

    predictions = model.predict(padded)
    predictions_classes = np.argmax(predictions, axis=1)
    df['sentiment'] = label_encoder.inverse_transform(predictions_classes)

    confindence_scores = np.max(predictions, axis=1)
    df['confidence_score'] = confindence_scores
    average_confidence = round(float(np.mean(confindence_scores)), 4)

    result_filename = f'result_{app_code}.xlsx'
    result_path = os.path.join(RESULT_FOLDER, result_filename)
    df.to_excel(result_path, index=False)

    sentiment_counts = df['sentiment'].value_counts(normalize=True)*100
    sentiment_percentage = {
        'positif': round(sentiment_counts.get('positif', 0), 2),
        'negatif': round(sentiment_counts.get('negatif', 0), 2),
        'netral': round(sentiment_counts.get('netral', 0), 2)
    }

    wordclod_filename = f'wordcloud_{app_code}_{uuid.uuid4().hex}.png'
    all_text = ' '.join(df['stemmed'])
    wordcloud_path = os.path.join('static', wordclod_filename)
    generate_wordcloud(all_text, wordcloud_path)

    return render_template(
        'outputPage.html',
        percentages=sentiment_percentage,
        download_link=result_filename,
        jumlah_data=jumlah_data,
        wordcloud_path=f'static/{wordclod_filename}',
        average_confidence=average_confidence,
    )

def generate_wordcloud(text, save_path):
    wordcloud = WordCloud(
        width=400,
        height=200,
        background_color='white',
    ).generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig(save_path, format='png')
    plt.close()

if __name__ == '__main__':
    app.run(debug=True)
