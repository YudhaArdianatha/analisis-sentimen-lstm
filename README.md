# Sentiment Analysis of Google Play Store App Reviews Using LSTM

This project implements a **Long Short-Term Memory (LSTM)**-based sentiment analysis model to classify app reviews from the Google Play Store into three categories: **positive**, **negative**, and **neutral**.  
The application is built with **Flask** as a web interface and supports direct text input, Excel file uploads, and fetching reviews directly from the Google Play Store.

---

## 📌 Overview

In today's digital era, user reviews of mobile applications are an important source of information for evaluation and development.  
However, the large volume of reviews, informal language, and unstructured sentences make manual analysis challenging.  
This project leverages LSTM to analyze Indonesian-language reviews, achieving an accuracy of **87.21%**.

**Key Features:**

- **Single text input** for quick analysis.
- **Excel file upload** with a `content` column for batch analysis.
- **Fetch Google Play Store reviews** using `google-play-scraper`.
- **Visualization** in the form of pie charts and word clouds.
- **Export analysis results** to Excel.

---

## 🛠 Technologies Used

- **Python 3.11**
- **Flask**
- **TensorFlow / Keras**
- **Pandas, NumPy**
- **Matplotlib**
- **WordCloud**
- **google-play-scraper**
- **nltk**
- **Sastrawi** (for Indonesian stemming)

---

## 🚀 How to Run

1. **Clone the repository**
   ```bash
   git clone https://github.com/YudhaArdianatha/analisis-sentimen-lstm.git
   cd analisis-sentimen-lstm
   ```
2. **Create and activate a virtual environment**
   ```bash
   python -m venv venv
   ```
   - **Activate the virtual environment:**
     - Mac/Linux:
       ```bash
       source venv/bin/activate
       ```
     - Windows:
       ```bash
       venv\Scripts\activate
       ```
3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
4. **Ensure the model/ folder contains:**
   - `best_lstm_modelV2.h5`
   - `tokenizer_lstmV2.pickle`
   - `label_encoder_lstmV2.pickle`
5. **Run the application**
   ```bash
   python app.py
   ```

## 📊 Dataset & Model Training

- Dataset: 7,235 reviews (3,606 negative, 2,501 positive, 1,128 neutral)
- Source: Google Play Store (google-play-scraper)
- Labeling: Manual
- Preprocessing:
  - Text cleaning
  - Case folding
  - Slang word normalization
  - Tokenization
  - Stopword removal
  - Stemming(sastrawi)
- Model : Long Short-Term Memory (LSTM)
- Accuracy: 87.21%
- Training Environment: Google Colab

## 📄 License

This project is created for educational and research purposes.
Feel free to use, modify, and improve it as needed.
