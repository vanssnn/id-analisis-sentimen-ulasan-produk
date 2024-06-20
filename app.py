# Import libraries
import streamlit as st
import pickle, json

import nltk
nltk.download('punkt')

from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# Set page config
st.set_page_config(
    page_title="ID - Analisis Sentimen Ulasan Produk",
    page_icon="📦",
)

with open('./utils/utils.json', 'r') as f:
    utils = json.load(f)

# Define the preprocess_text function
def preprocess_text(sentence):
    stemmer = StemmerFactory().create_stemmer()

    word_list = word_tokenize(sentence.lower()) 
    word_list = [word for word in word_list if word not in utils['punctuation']]
    word_list = [word for word in word_list if ((word not in utils['num_words']) and (word.isalpha()))]
    word_list = [stemmer.stem(word) for word in word_list if word not in utils['stopwords']]
    return ' '.join(word_list)

# Set title and description
st.title('🇮🇩 -📦Analisis Sentimen Ulasan Produk')
st.markdown("**Aplikasi Web** ini bertujuan untuk menganalisis sentimen ulasan produk berdasarkan teks yang dimasukkan oleh pengguna. Model akan memprediksi ulasan tersebut dan mengkategorikannya menjadi label **Positif** atau **Negatif**.")
st.markdown('Cek [GitHub repository](%s) untuk analisa dari model yang digunakan.' % 'https://github.com/vanssnn/id-analisis-sentimen-ulasan-produk')

# Widget to get the user input
st.header("👇Masukkan Ulasan Produk")
with st.form("my_form"):
    sentence = st.text_area("Tulis ulasan produk dalam text box berikut:", max_chars=1000, height=200)
    predict_btn = st.form_submit_button("Prediksi Sentimen")

if predict_btn:
    if sentence != '':
        # Preprocess the text
        sentence = preprocess_text(sentence)

        # Load the model and vectorizer
        with open('./utils/model.pickle', 'rb') as f:
            model = pickle.load(f)
        
        with open('./utils/vectorizer.pickle', 'rb') as f:
            vect = pickle.load(f)

        # Predict the sentiment
        sentence_vect = vect.transform([sentence])
        sentence_vect = sentence_vect.toarray()
        prediction = model.predict(sentence_vect)

        # Display the prediction
        if prediction[0] == 1:
            st.success("👍Sentimen: **Positif**")
        else:
            st.error("👎Sentimen: **Negatif**")
    else:
        st.warning("⚠️Masukkan ulasan produk terlebih dahulu!")