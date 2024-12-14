import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
from PIL import Image
from streamlit_option_menu import option_menu
import requests

# ===== KONFIGURASI API GROQ =====
GROQ_API_KEY = "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"  # GANTI DENGAN API KEY GROQ ANDA
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

# Label klasifikasi sampah
label_models = [
    "battery", "biological", "brown-glass", "cardboard", "clothes",
    "green-glass", "metal", "paper", "plastic", "shoes", "trash", "white-glass"
]

# Informasi dan kategori sampah
label_information = {
    "battery": {
        "deskripsi": "Baterai bekas termasuk sampah elektronik yang mengandung bahan kimia berbahaya seperti timbal dan merkuri.",
        "penanganan": "Baterai bekas harus dikumpulkan dan didaur ulang melalui pusat daur ulang elektronik.",
        "kategori": "E-Waste"
    },
    "biological": {
        "deskripsi": "Sampah biologis berasal dari sisa makhluk hidup seperti sisa makanan dan daun-daunan.",
        "penanganan": "Sampah ini dapat diolah menjadi kompos untuk pupuk alami.",
        "kategori": "Organik"
    },
    "brown-glass": {
        "deskripsi": "Sampah kaca berwarna coklat seperti botol minuman bekas.",
        "penanganan": "Pisahkan kaca berwarna dari jenis kaca lain dan kirimkan ke pusat daur ulang kaca.",
        "kategori": "Glass"
    },
    "cardboard": {
        "deskripsi": "Kardus atau kertas tebal bekas yang umum digunakan sebagai kemasan.",
        "penanganan": "Lipat dan kumpulkan kardus untuk didaur ulang menjadi produk kertas baru.",
        "kategori": "Paper"
    },
    "clothes": {
        "deskripsi": "Pakaian bekas yang sudah tidak digunakan.",
        "penanganan": "Sumbangkan pakaian layak pakai atau gunakan kembali sebagai kain lap.",
        "kategori": "Reusable"
    },
    "green-glass": {
        "deskripsi": "Sampah kaca berwarna hijau seperti botol minuman.",
        "penanganan": "Pisahkan dan daur ulang bersama kaca berwarna lainnya.",
        "kategori": "Glass"
    },
    "metal": {
        "deskripsi": "Logam seperti kaleng minuman, besi tua, atau aluminium.",
        "penanganan": "Logam dapat dilebur kembali dan digunakan untuk pembuatan produk baru.",
        "kategori": "Metal"
    },
    "paper": {
        "deskripsi": "Sampah kertas seperti koran, majalah, atau kertas bekas.",
        "penanganan": "Kumpulkan dan daur ulang menjadi kertas daur ulang.",
        "kategori": "Paper"
    },
    "plastic": {
        "deskripsi": "Sampah plastik termasuk botol, kantong plastik, dan sedotan.",
        "penanganan": "Pisahkan plastik berdasarkan jenisnya dan kirim ke fasilitas daur ulang.",
        "kategori": "Plastic"
    },
    "shoes": {
        "deskripsi": "Sepatu bekas yang sudah tidak layak digunakan.",
        "penanganan": "Sepatu bekas dapat disumbangkan atau didaur ulang menjadi bahan lain.",
        "kategori": "Reusable"
    },
    "trash": {
        "deskripsi": "Sampah umum yang tidak dapat didaur ulang atau digunakan kembali.",
        "penanganan": "Buang ke tempat sampah akhir atau gunakan pengelolaan sampah terorganisir.",
        "kategori": "Mixed Waste"
    },
    "white-glass": {
        "deskripsi": "Sampah kaca bening seperti botol kaca putih atau gelas.",
        "penanganan": "Pisahkan kaca bening dan kirim ke pusat daur ulang kaca.",
        "kategori": "Glass"
    }
}

# ===== FUNGSI CHATBOT =====
def get_groq_response(prompt):
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "llama3-8b-8192",  # Model Groq yang digunakan
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7
    }
    try:
        response = requests.post(GROQ_API_URL, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Terjadi kesalahan: {e}"

with st.sidebar:
    selected = option_menu(
        menu_title="Main Menu",
        options=["Home", "Chatbot"],
        icons=["house", "robot"],
        menu_icon="cast",
        default_index=0
    )

# Muat model klasifikasi sampah
model = tf.keras.models.load_model("./final_garbage_classification_model.keras")
if selected == "Home":
    # Konfigurasi Streamlit
    st.title("Aplikasi Klasifikasi Sampah")

    confidence_threshold = 0.9

    uploaded_file = st.file_uploader("Unggah Gambar Sampah:", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        # Proses gambar
        image = Image.open(uploaded_file)
        st.image(image, caption="Gambar yang Diunggah", width=200)

        # Preprocessing gambar
        image = np.array(image)
        image = cv2.resize(image, (224, 224))
        image = np.expand_dims(image, axis=0)
        image = tf.keras.applications.mobilenet_v2.preprocess_input(image)

        # Prediksi
        pred = model.predict(image)
        confidence = np.max(pred)
        predicted_class = np.argmax(pred)

        # Logika threshold
        if confidence < confidence_threshold:
            output = "Bukan Sampah"
            kategori = "-"
            deskripsi = "Gambar tidak dikenali atau bukan sampah yang valid."
            penanganan = "-"
        else:
            output = label_models[predicted_class]
            kategori = label_information[output]["kategori"]
            deskripsi = label_information[output]["deskripsi"]
            penanganan = label_information[output]["penanganan"]

        # Tampilkan output
        st.write(f"### Prediksi: {output}")
        st.write(f"**Kategori Sampah**: {kategori}")
        st.write(f"**Deskripsi**: {deskripsi}")
        st.write(f"**Penanganan**: {penanganan}")

    else:
        st.info("Silakan unggah gambar sampah untuk diklasifikasikan.")

# ===== HALAMAN CHATBOT =====
if selected == "Chatbot":
    st.title("🤖 Chatbot Edukasi Sampah AI (Groq)")
    st.write("Ketik pertanyaan Anda seputar sampah dan pengelolaannya:")

    user_input = st.text_area("Masukkan pertanyaan Anda di sini:", height=100)
    if st.button("Kirim Pertanyaan"):
        if user_input:
            with st.spinner("Sedang mencari jawaban..."):
                response = get_groq_response(user_input)
            st.subheader("Jawaban:")
            st.write(response)
        else:
            st.warning("Silakan ketik sesuatu untuk ditanyakan!")
