from flask import Flask, render_template, request
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from preprocessing.preprocessing import cleaningText, casefoldingText, fix_slangwords, tokenizingText, filteringText, stemmingText



app = Flask(__name__)

model = load_model('model/best_lstm_modelV2.h5')

with open('model/tokenizer_lstmV2.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

with open('model/label_encoder_lstmV2.pickle', 'rb') as handle:
    label_encoder = pickle.load(handle)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/outputPage', methods=['GET', 'POST'])
def output_page():
    if request.method == 'POST':
        sentiment = request.form['sentiment']

        raw = sentiment
        cleaned = cleaningText(raw)
        casefolded = casefoldingText(cleaned)
        fixed_slang = fix_slangwords(casefolded)
        tokenized = tokenizingText(fixed_slang)
        filtered = filteringText(tokenized)
        stemmed = stemmingText(filtered)

        sequence = tokenizer.texts_to_sequences([stemmed])
        padded_sequence = pad_sequences(sequence, maxlen=200)

        prediction = model.predict(padded_sequence)
        prediction_label = np.argmax(prediction, axis=1)
        decoded_label = label_encoder.inverse_transform(prediction_label)

        return render_template('outputPage.html',
                               sentiment=decoded_label[0],
                            #    raw=raw,
                            #    cleaned=cleaned,
                            #    casefolded=casefolded,
                            #    fixed_slang=fixed_slang,
                            #    tokenized=tokenized,
                            #    filtered=filtered,
                            #    stemmed=stemmed,
                               )
    return redirect('/')



if __name__ == '__main__':
    app.run(debug=True)
