import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import time

# Memuat model .keras
model_path = "best_model_1.keras"  # Ganti dengan path model Anda
model = tf.keras.models.load_model(model_path)

# Nama kelas dari model (misal, motif batik)
class_labels = [
    "Insang",
    "Kawung",
    "Megamendung",
    "Parang",
    "Sidoluhur",
    "Truntum",
    "Tumpal",
]

# Informasi tambahan untuk setiap motif batik
batik_info = {
    "Insang": {
        "asal": "Solo, Jawa Tengah",
        "sejarah": "Motif Insang terinspirasi oleh bentuk insang ikan, yang melambangkan kesuburan dan kehidupan.",
        "arti": "Melambangkan kehidupan, kesuburan, dan ketenangan.",
    },
    "Kawung": {
        "asal": "Yogyakarta",
        "sejarah": "Motif Kawung merupakan salah satu motif batik tertua yang menggambarkan pola geometris dari bunga teratai.",
        "arti": "Melambangkan kesucian, keselarasan, dan keseimbangan.",
    },
    "Megamendung": {
        "asal": "Cirebon, Jawa Barat",
        "sejarah": "Motif Megamendung terinspirasi dari awan mendung di langit yang melambangkan ketenangan dan perlindungan.",
        "arti": "Melambangkan kedamaian, ketenangan, dan perlindungan Tuhan.",
    },
    "Parang": {
        "asal": "Yogyakarta",
        "sejarah": "Motif Parang adalah motif batik yang menggambarkan garis diagonal yang terhubung, simbol dari perjuangan dan keteguhan hati.",
        "arti": "Melambangkan kekuatan, perjuangan, dan keteguhan hati.",
    },
    "Sidoluhur": {
        "asal": "Solo, Jawa Tengah",
        "sejarah": "Motif Sidoluhur menggambarkan pola segi empat dan persegi yang melambangkan keharmonisan kehidupan.",
        "arti": "Melambangkan keharmonisan dan kebahagiaan dalam kehidupan.",
    },
    "Truntum": {
        "asal": "Yogyakarta",
        "sejarah": "Motif Truntum berasal dari kata 'truntum' yang berarti tumbuh atau berkembang, melambangkan cinta yang berkembang.",
        "arti": "Melambangkan cinta, kedamaian, dan kasih sayang.",
    },
    "Tumpal": {
        "asal": "Solo, Jawa Tengah",
        "sejarah": "Motif Tumpal menggambarkan tumpalan atau titik yang terpusat, menggambarkan kedekatan dengan Tuhan.",
        "arti": "Melambangkan kedekatan dengan Tuhan dan pengorbanan.",
    },
}

# Ambang batas untuk kepercayaan prediksi
confidence_threshold = 50  # Threshold dalam persentase


# Fungsi untuk memproses gambar yang diunggah untuk prediksi
def preprocess_image(img_file):
    img = image.load_img(
        img_file, target_size=(224, 224)
    )  # Ubah ukuran gambar sesuai input model
    img_array = image.img_to_array(img)  # Mengonversi gambar menjadi array
    img_array = np.expand_dims(img_array, axis=0)  # Menambahkan dimensi batch
    img_array = tf.keras.applications.resnet50.preprocess_input(
        img_array
    )  # Preprocessing sesuai dengan model
    return img_array


# Fungsi untuk memprediksi gambar
def predict_image(img_file):
    # Proses gambar
    img_array = preprocess_image(img_file)

    # Prediksi menggunakan model
    predictions = model.predict(img_array)

    # Menentukan kelas yang diprediksi dan kepercayaan (confidence)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_label = class_labels[predicted_class_index]
    confidence = np.max(predictions, axis=1)[0] * 100

    # Menentukan apakah prediksi berada di bawah threshold
    if confidence < confidence_threshold:
        predicted_class_label = (
            "Bukan Batik atau motif batik tidak dikenali/belum dilatih"
        )
        confidence = 0.0  # Tidak ada kepercayaan karena tidak dapat dikenali
    return predicted_class_label, confidence


# Menggunakan Streamlit untuk upload gambar atau ambil gambar dari kamera
st.title("Aplikasi Klasifikasi Motif Batik")

input_choice = st.radio(
    "Pilih Metode Input",
    ["Ambil Gambar dari Kamera", "Unggah Gambar dari Perangkat"],
)

if input_choice == "Ambil Gambar dari Kamera":
    camera_input = st.camera_input("Ambil gambar untuk diprediksi")
    if camera_input is not None:
        # Menampilkan gambar yang diambil langsung
        img = Image.open(camera_input)
        st.image(
            img, caption="Gambar yang diambil", use_container_width=True, width=300
        )

        # Melakukan prediksi dan menampilkan hasil
        predicted_class_label, confidence = predict_image(camera_input)

        # Menampilkan prediksi yang telah dilakukan
        st.write(f"**Nama Motif Batik**: {predicted_class_label}")
        st.write(f"**Kepercayaan**: {confidence:.2f}%")

        # Menampilkan informasi lebih lanjut tentang motif batik
        batik_info_details = batik_info.get(predicted_class_label, {})
        if batik_info_details:
            st.write(f"**Asal**: {batik_info_details['asal']}")
            st.write(f"**Sejarah**: {batik_info_details['sejarah']}")
            st.write(f"**Arti Motif**: {batik_info_details['arti']}")

elif input_choice == "Unggah Gambar dari Perangkat":
    uploaded_file = st.file_uploader(
        "Pilih Gambar Batik untuk Klasifikasi", type=["jpg", "jpeg", "png"]
    )
    if uploaded_file is not None:
        # Menampilkan gambar yang diunggah hanya sekali
        img = Image.open(uploaded_file)
        st.image(
            img, caption="Gambar yang diunggah", use_container_width=True, width=300
        )

        # Melakukan prediksi dan menampilkan hasil hanya sekali
        with st.spinner("Memproses gambar..."):
            predicted_class_label, confidence = predict_image(uploaded_file)
            time.sleep(3)  # Simulasi delay untuk menunjukkan loading
        st.success("Prediksi selesai!")

        # Menampilkan prediksi yang telah dilakukan
        st.write(f"**Nama Motif Batik**: {predicted_class_label}")
        st.write(f"**Kepercayaan**: {confidence:.2f}%")

        # Menampilkan informasi lebih lanjut tentang motif batik
        batik_info_details = batik_info.get(predicted_class_label, {})
        if batik_info_details:
            st.write(f"**Asal**: {batik_info_details['asal']}")
            st.write(f"**Sejarah**: {batik_info_details['sejarah']}")
            st.write(f"**Arti Motif**: {batik_info_details['arti']}")

else:
    st.write(
        "Silakan pilih salah satu metode input untuk mengunggah atau mengambil gambar."
    )
