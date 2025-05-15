import re
import string
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import json
import os


def cleaningText(text):
    text = re.sub(r'@[A-Za-z0-9]+', '', text) # remove mentions
    text = re.sub(r'#[A-Za-z0-9]+', '', text) # remove hashtag
    text = re.sub(r'RT[\s]+', '', text)       # remove RT
    text = re.sub(r"http\S+", '', text)       # remove link
    text = re.sub(r'[0-9]+', '', text)        # remove numbers

    # Ganti tanda baca dengan spasi
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)

    # Ganti newline dengan spasi
    text = text.replace('\n', ' ')

    # Hapus spasi berlebih
    text = re.sub('\s+', ' ', text)

    # Hapus spasi di awal dan akhir
    text = text.strip()

    return text


def casefoldingText(text): # Converting all the characters in a text into lower case
    text = text.lower()
    return text

def tokenizingText(text): # Tokenizing or splitting a string, text into a list of tokens
    text = word_tokenize(text)
    return text

# Definisi kata-kata penting untuk sentimen yang tidak boleh dihapus
additional_important_words = ['buruk', 'jelek', 'bagus', 'keren', 'mantap', 'mengecewakan',
                              'banget', 'sangat', 'sekali', 'tidak', 'jangan', 'benci', 'suka',
                              'senang', 'sedih', 'marah', 'kesal', 'parah', 'puas', 'kecewa'
                              ,'gak', 'iya', 'baik', 'biasa', 'ok', 'lumayan']

def filteringText(text): # Remove stopwords in a text
    listStopwords = set(stopwords.words('indonesian'))
    listStopwords1 = set(stopwords.words('english'))
    listStopwords.update(listStopwords1)
    listStopwords.update(['yaa','nya','na','sih','ku',"di","ga","ya","gaa","loh","kah","woi","woii","woy"])

    # Jangan filter kata-kata penting untuk sentimen
    for word in additional_important_words:
        if word in listStopwords:
            listStopwords.remove(word)

    filtered = []
    for txt in text:
        if txt not in listStopwords:
            filtered.append(txt)
    text = filtered
    return text

def stemmingText(text): # Reducing a word to its word stem that affixes to suffixes and prefixes
    # Membuat objek stemmer
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()

    # Memecah teks menjadi daftar kata
    if isinstance(text, str):
        words = text.split()
    else:
        # Asumsikan input adalah list
        words = text

    # Menerapkan stemming pada setiap kata dalam daftar
    stemmed_words = [stemmer.stem(word) for word in words]

    # Menggabungkan kata-kata yang telah distem
    stemmed_text = ' '.join(stemmed_words)

    return stemmed_text


with open(os.path.join(os.path.dirname(__file__), 'slangwords.json'), 'r', encoding='utf-8') as f:
    slangwords = json.load(f)

def fix_slangwords(text):
    words = text.split()
    fixed_words = []

    for word in words:
        if word.lower() in slangwords:
            fixed_words.append(slangwords[word.lower()])
        else:
            fixed_words.append(word)

    fixed_text = ' '.join(fixed_words)
    return fixed_text