import streamlit as st
import tensorflow as tf
from streamlit_option_menu import option_menu
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import time
import io

# Backgorund
page_bg_img = """
<style>
[data-testid="stAppViewContainer"] {
    background-image: url("https://images.ctfassets.net/nnkxuzam4k38/2SvDjcgyav5C1DOb79JKXl/d3b06db5bb6bdb4ab237f666b5b4980e/compute-ea4c57a4.png");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}
[data-testid="stHeader"] {
    background-color: rgba(0, 0, 0, 0);
}
[data-testid="stSidebarContent"] {
    background-color: rgba(0 ,0, 0, 0);
}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

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
        "asal": "Pontianak, Kalimantan Barat.",
        "ciri": "Batik ini menonjolkan motif yang terinspirasi dari bentuk insang ikan, dengan pola yang terlihat seperti garis-garis atau gelombang yang mengingatkan pada struktur insang. Warna-warna yang digunakan cenderung cerah dan segar, seperti biru, hijau, coklat, dan putih, menggambarkan hubungan yang kuat dengan alam dan laut.",
        "sejarah": "Batik Insang berasal dari Pontianak, Kalimantan Barat, dan memiliki hubungan erat dengan budaya lokal serta kehidupan masyarakat sekitar. Nama Insang merujuk pada motif yang menyerupai struktur insang ikan, menggambarkan hubungan masyarakat dengan alam, terutama perairan dan laut di Kalimantan Barat. Batik ini dipengaruhi oleh kekayaan alam daerah, dengan motif yang sering menggambarkan ikan, bunga, dan elemen alam lainnya. Awalnya, batik ini digunakan dalam upacara adat dan kehidupan sehari-hari, namun seiring waktu, Batik Insang dikenal lebih luas dan menjadi simbol kebanggaan masyarakat Pontianak. Selain menjadi produk kerajinan lokal, Batik Insang juga berkembang menjadi komoditas populer di pasar lokal dan nasional, dan meskipun sudah modern, tetap mempertahankan nilai budaya yang terkandung di dalamnya.",
        "arti": "Secara simbolis, batik ini juga mencerminkan ketahanan dan kelangsungan hidup, karena insang ikan merupakan alat vital yang mendukung kehidupan ikan di air. Dengan demikian, Batik Insang dapat dianggap sebagai simbol kesuburan alam, kehidupan yang berkelanjutan, dan hubungan harmonis antara manusia dengan alam.",
    },
    "Kawung": {
        "asal": "Yogyakarta dan Solo.",
        "ciri": "Batik Kawung memiliki ciri khas motif berbentuk lingkaran atau oval yang teratur dan berulang, menciptakan pola simetris yang harmonis. Motif ini sering kali menyerupai bunga atau buah kawung (kolang-kaling). Warna yang digunakan umumnya cenderung gelap atau netral seperti coklat, hitam, dan biru tua.",
        "sejarah": "Motif ini pertama kali berkembang pada masa kerajaan Mataram di Jawa, sekitar abad ke-18. Batik Kawung pada awalnya digunakan oleh kalangan kerajaan dan aristokrat sebagai simbol status sosial dan identitas budaya. Meskipun Batik Kawung sempat meredup pada masa penjajahan, motif ini kembali populer pada abad ke-20 berkat upaya pelestarian dan kebangkitan industri batik di Indonesia. Kini, Batik Kawung menjadi salah satu motif batik yang paling dikenal dan sering digunakan dalam pakaian adat serta acara formal.",
        "arti": "Motif Kawung menggambarkan pola lingkaran atau oval yang teratur dan berulang, yang melambangkan kesempurnaan, keseimbangan, dan kehidupan yang berputar. Batik ini sangat dihargai karena simetri dan keindahan desainnya, yang mencerminkan filosofi hidup masyarakat Jawa yang mementingkan keharmonisan.",
    },
    "Megamendung": {
        "asal": "Cirebon, Jawa Barat.",
        "ciri": "Batik Megamendung memiliki motif utama yang menyerupai awan besar atau mendung, dengan bentuk simetris dan teratur, sering kali menggambarkan pola bergelombang atau melengkung. Motif ini menggunakan warna-warna cerah dan kontras seperti biru, merah, hijau, kuning, dan ungu, yang menciptakan kesan dinamis dan mencolok. ",
        "sejarah": "Batik Megamendung memiliki sejarah yang kuat terkait dengan pengaruh budaya lokal serta budaya Tionghoa. Batik ini pertama kali berkembang di Cirebon pada masa Kesultanan Cirebon, yang dikenal sebagai pusat perdagangan dan pertemuan berbagai budaya, termasuk budaya Tionghoa. Pada awalnya, Batik Megamendung digunakan oleh kalangan istana dan masyarakat Cirebon sebagai simbol kebanggaan dan identitas budaya. Seiring berjalannya waktu, batik ini mulai dikenal lebih luas dan menjadi salah satu motif batik yang khas dari Cirebon.",
        "arti": "Batik Megamendung memiliki arti yang berkaitan dengan simbolisme cuaca dan kehidupan. Mega dalam bahasa Indonesia berarti awan, sementara mendung merujuk pada awan gelap sebelum hujan. Secara keseluruhan, Batik Megamendung melambangkan perubahan, kehidupan yang dinamis, serta harapan akan hujan yang membawa berkah. Awan dalam motif ini juga dianggap sebagai simbol keberuntungan dan kemakmuran, yang mengacu pada budaya Tionghoa yang memandang awan sebagai simbol positif. Dengan demikian, Batik Megamendung menggambarkan siklus kehidupan, transformasi, dan harapan akan masa depan yang lebih baik.",
    },
    "Parang": {
        "asal": "Yogyakarta dan Solo.",
        "ciri": "Motif Batik Parang terdiri dari pola yang berulang dan simetris, dengan garis-garis yang membentuk bentuk mirip parang (senjata tradisional). Motif ini terlihat seperti garis diagonal atau zig-zag yang terhubung dengan pola lain, membentuk kesan yang sangat teratur dan dinamis.",
        "sejarah": "Motif Batik Parang pertama kali berkembang di kalangan keluarga kerajaan dan bangsawan di Yogyakarta dan Surakarta. Awalnya, Batik Parang digunakan oleh kalangan kerajaan dan aristokrat Jawa untuk berbagai keperluan resmi, seperti upacara adat, pernikahan, dan acara kerajaan. Batik ini digunakan sebagai simbol kehormatan dan status sosial. Seiring berjalannya waktu, Batik Parang mulai dikenal lebih luas oleh masyarakat umum dan digunakan dalam berbagai kesempatan, termasuk dalam pakaian sehari-hari. Batik Parang sempat mengalami kemunduran pada masa penjajahan Belanda, tetapi bangkit kembali setelah kemerdekaan Indonesia. Pada abad ke-20, Batik Parang mengalami kebangkitan dan menjadi salah satu motif batik yang sangat dihargai dan dilestarikan. ",
        "arti": "Batik Parang melambangkan semangat perjuangan, ketangguhan, dan keberanian dalam menghadapi rintangan. Motif ini juga mencerminkan nilai-nilai kehormatan, kesetiaan, dan pengabdian dalam budaya Jawa. Sebagai salah satu motif batik yang paling tua, Batik Parang menjadi simbol dari kekuatan spiritual dan fisik, serta karakter yang gigih dan tidak mudah menyerah.",
    },
    "Sidoluhur": {
        "asal": "Keraton khususnya di Yogyakarta dan Surakarta.",
        "ciri": "Motif Batik Sidoluhur biasanya terdiri dari pola geometris yang simetris, sering kali berupa garis-garis atau bentuk-bentuk yang teratur. Warna yang digunakan dalam Batik Sidoluhur cenderung lembut dan elegan, seperti warna coklat, krem, dan putih. ",
        "sejarah": "Batik Sidoluhur pertama kali dikenal pada masa kerajaan Mataram, terutama di lingkungan keraton Yogyakarta. Sejak zaman dahulu, batik digunakan oleh keluarga kerajaan dan masyarakat bangsawan dalam berbagai acara adat, seperti pernikahan, upacara keagamaan, dan acara resmi lainnya.",
        "arti": "Batik Sidoluhur menggambarkan harapan agar kehidupan seseorang menjadi lebih baik, mulia, dan berhasil. Filosofi ini terkait erat dengan keyakinan dalam budaya Jawa yang sangat menghargai nilai-nilai kehormatan, kebahagiaan, dan kesuksesan dalam hidup. Batik Sidoluhur juga dipercaya dapat membawa berkah dan memperkuat hubungan antar sesama, terutama dalam konteks pernikahan atau acara yang melibatkan ikatan sosial yang penting.",
    },
    "Truntum": {
        "asal": "Surakarta, Jawa Tengah.",
        "ciri": "Ciri khas batik Truntum adalah motifnya yang menyerupai kuntum (bunga) atau kembang di langit, khususnya bunga tanjung. Pemilihan warna pada batik Truntum cenderung menggunakan nuansa alam, seperti coklat, biru tua, dan hitam. Desain batik Truntum umumnya cenderung sederhana dan simetris, dengan motif bunga dan pola geometris yang diatur dengan rapi.",
        "sejarah": "Batik Truntum merupakan motif batik yang diciptakan oleh Kanjeng Ratu Kencana, atau Permaisuri Ingkang Sinuhun Sri Susuhunan Pakubuwana III dari Surakarta. Dalam sejarahnya, motif ini lahir karena sang permaisuri, yang dikenal sebagai Ratu Beruk, tidak dapat memberikan keturunan kepada Pakubuwono III, sehingga sang raja memutuskan untuk menikah lagi. Ratu Beruk yang merasa tak berdaya hanya bisa menerima keputusan tersebut. Dalam kesedihannya, ia merenung dengan menatap bintang-bintang di langit malam, yang selama ini menemani kesendirian dan kesepiannya. Dari perenungannya itu, ia terinspirasi untuk menciptakan motif batik yang menggambarkan bintang-bintang di langit malam, serta bunga kuntum atau kembang yang tampak seperti mekar di angkasa.",
        "arti": "Batik Truntum adalah motif batik Jawa yang memiliki makna cinta kasih yang tumbuh kembali, abadi, dan semakin subur. Motif ini sering dipakai oleh orang tua pengantin pada hari pernikahan, dengan harapan cinta kasih akan menghinggapi kedua mempelai. ",
    },
    "Tumpal": {
        "asal": "Batik motif tumpal berasal dari berbagai daerah di Indonesia, tetapi memiliki hubungan kuat dengan batik Betawi.",
        "ciri": "Motif tumpal umumnya memiliki bentuk dasar segitiga sama kaki atau segitiga lancip. Bentuk-bentuk segitiga ini kemudian disusun secara berulang untuk menciptakan pola hias tertentu. Ciri khas dari Batik Tumpal adalah pola segitiga yang disebut tumpal, yang sering kali disusun tegak, dengan segitiga yang menghadap ke atas atau ke bawah. Pola ini biasanya ditempatkan secara berulang pada kain, dengan tumpal sering muncul di bagian pinggir atau ujung kain.",
        "sejarah": "Suwati Kartiwa dalam tulisannya yang berjudul “Batik Betawi: Dalam Perspektif Budaya Kreatif” menyatakan bahwa kemungkinan besar asal usul Batik Betawi berakar dari asimilasi antara masyarakat Jawa, khususnya penghasil batik dari daerah pesisir, dengan masyarakat Betawi yang telah lama tinggal dan berinteraksi dalam lingkungan Kota Batavia. Proses interaksi ini kemudian menyebarkan budaya mereka dalam bentuk kain batik. Seiring berjalannya waktu, Batik Betawi berkembang dengan menonjolkan motif tumpal, yang memiliki bentuk geometris segitiga, yang menjadi elemen penting dan harus ada pada bagian depan kain batik tersebut. Motif tumpal ini menjadi salah satu ciri khas dalam batik Betawi.",
        "arti": "Motif batik tumpal secara umum bermakna sebagai penolak bala atau penjauh bencana. Bentuk segitiga tumpal yang runcing, menyerupai gigi buaya, dianggap memiliki kekuatan magis untuk melindungi pemakainya. Selain itu, motif tumpal juga memiliki makna filosofis yang berkaitan dengan keselarasan antara manusia, semesta, dan Tuhan. ",
    },
}

# Ambang batas untuk kepercayaan Klasifikasi
confidence_threshold = 75  # Threshold dalam persentase


# Fungsi untuk memproses gambar yang diunggah untuk Klasifikasi
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


# Fungsi untuk memKlasifikasi gambar
def predict_image(img_file):
    # Proses gambar
    img_array = preprocess_image(img_file)

    # Klasifikasi menggunakan model
    predictions = model.predict(img_array)

    # Menentukan kelas yang diKlasifikasi dan kepercayaan (confidence)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_label = class_labels[predicted_class_index]
    confidence = np.max(predictions, axis=1)[0] * 100

    # Menentukan apakah Klasifikasi berada di bawah threshold
    if confidence < confidence_threshold:
        predicted_class_label = (
            "Bukan Batik atau motif batik tidak dikenali/belum dilatih"
        )
        confidence = 0.0  # Tidak ada kepercayaan karena tidak dapat dikenali
    return predicted_class_label, confidence


# Menyimpan riwayat ke session state jika belum ada
if "history" not in st.session_state:
    st.session_state.history = []

from PIL import Image


# Halaman Boarding/Home
def show_home_page():
    st.title("Selamat datang di Aplikasi Klasifikasi Motif Batik")

    st.subheader(
        "Temukan informasi tentang motif batik tradisional dengan mudah menggunakan aplikasi ini!"
    )
    st.write(
        """Dibekali model CNN ResNet-50, aplikasi ini memungkinkan pengguna untuk mengenali motif batik hanya dengan mengambil gambar langsung melalui kamera atau mengunggahnya. 
        Dapatkan klasifikasi motif batik yang terdeteksi secara akurat, sehingga Anda dapat lebih memahami dan menghargai warisan budaya Indonesia!"
        """
    )

    # Pengertian umum Batik
    st.subheader("Pengertian Batik Tradisional")
    st.write(
        """Batik adalah salah satu seni dan budaya tradisional Indonesia yang telah diakui oleh UNESCO sebagai warisan budaya dunia. 
        Batik memiliki ciri khas yaitu proses pewarnaan kain dengan menggunakan lilin untuk menghasilkan motif yang sangat beragam.
        Motif batik dipengaruhi oleh kebudayaan lokal, alam sekitar, dan nilai-nilai filosofi yang dalam.
        """
    )

    # Kelas-kelas Batik yang Dapat Diklasifikasikan
    st.subheader("Kelas-kelas Batik yang Dapat Dikenali")
    st.write(
        """Aplikasi ini dapat mengklasifikasikan 7 jenis motif batik, yang masing-masing memiliki ciri khas dan filosofi yang mendalam. 
        Berikut adalah 7 kelas motif batik yang dapat dikenali oleh aplikasi ini:
        """
    )

    # Menampilkan gambar contoh motif batik dalam dua kolom per baris
    st.write("### Contoh Gambar Motif Batik Yang Dapat Dikenali")

    class_labels = [
        "Insang",
        "Kawung",
        "Megamendung",
        "Parang",
        "Sidoluhur",
        "Truntum",
        "Tumpal",
    ]

    image_paths = [
        "images/insang.jpg",  # Ganti dengan path gambar asli
        "images/kawung.jpg",  # Ganti dengan path gambar asli
        "images/megamendung.jpg",  # Ganti dengan path gambar asli
        "images/parang.jpg",  # Ganti dengan path gambar asli
        "images/sidoluhur.jpg",  # Ganti dengan path gambar asli
        "images/truntum.jpg",  # Ganti dengan path gambar asli
        "images/tumpal.jpg",  # Ganti dengan path gambar asli
    ]

    # Menampilkan gambar dalam dua kolom per baris dengan ukuran 300x300
    for i in range(0, len(class_labels), 2):
        col1, col2 = st.columns(2)  # Membuat dua kolom
        with col1:
            if i < len(class_labels):
                img = Image.open(image_paths[i])  # Membuka gambar dengan PIL
                img = img.resize((300, 300))  # Mengubah ukuran gambar menjadi 300x300
                st.image(img, caption=f"Motif Batik: {class_labels[i]}")
        with col2:
            if i + 1 < len(class_labels):
                img = Image.open(image_paths[i + 1])  # Membuka gambar dengan PIL
                img = img.resize((300, 300))  # Mengubah ukuran gambar menjadi 300x300
                st.image(img, caption=f"Motif Batik: {class_labels[i + 1]}")

    st.write(
        "Silakan masuk ke halaman Klasifikasi Batik dan pilih metode input gambar untuk mulai mengklasifikasikan motif batik!"
    )


# Halaman Klasifikasi
def show_classification_page():
    st.title("Aplikasi Klasifikasi Motif Batik")

    input_choice = st.radio(
        "Pilih Metode Input",
        ["Ambil Gambar dari Kamera", "Unggah Gambar dari Perangkat"],
        key="classification_input",
    )

    if input_choice == "Ambil Gambar dari Kamera":
        camera_input = st.camera_input("Ambil gambar untuk diKlasifikasi")
        if camera_input is not None:
            # Menampilkan gambar yang diambil langsung
            img = Image.open(camera_input)

            # Menempatkan gambar di tengah dengan menggunakan st.columns
            col1, col2, col3 = st.columns([1, 5, 1])  # Membuat tiga kolom
            with col2:  # Menempatkan gambar di kolom tengah
                st.image(img, caption="Gambar yang diambil", width=300)

            # Melakukan Klasifikasi dan menampilkan hasil hanya sekali
            with st.spinner("Memproses gambar..."):
                predicted_class_label, confidence = predict_image(camera_input)
                time.sleep(1)  # Simulasi delay untuk menunjukkan loading
            st.success("Klasifikasi selesai!")

            # Menampilkan Klasifikasi yang telah dilakukan
            st.write(f"**Nama Motif Batik**: {predicted_class_label}")
            st.write(f"**Kepercayaan**: {confidence:.2f}%")

            # Menampilkan informasi lebih lanjut tentang motif batik
            batik_info_details = batik_info.get(predicted_class_label, {})
            if batik_info_details:
                st.write(f"**Asal**: {batik_info_details['asal']}")
                st.write(f"**Ciri-Ciri**: {batik_info_details['ciri']}")
                st.write(f"**Sejarah**: {batik_info_details['sejarah']}")
                st.write(f"**Arti Motif**: {batik_info_details['arti']}")

            # Menyimpan gambar dan hasil Klasifikasi ke riwayat
            img_bytes = io.BytesIO()
            img.save(img_bytes, format="PNG")
            img_bytes = img_bytes.getvalue()
            st.session_state.history.append(
                {
                    "image": img_bytes,
                    "label": predicted_class_label,
                    "confidence": confidence,
                }
            )

    elif input_choice == "Unggah Gambar dari Perangkat":
        uploaded_file = st.file_uploader(
            "Pilih Gambar Batik untuk Klasifikasi",
            type=["jpg", "jpeg", "png"],
            key="upload_image",
        )
        if uploaded_file is not None:
            # Menampilkan gambar yang diunggah hanya sekali
            img = Image.open(uploaded_file)

            # Menempatkan gambar di tengah dengan menggunakan st.columns
            col1, col2, col3 = st.columns([1, 5, 1])  # Membuat tiga kolom
            with col2:  # Menempatkan gambar di kolom tengah
                st.image(img, caption="Gambar yang diunggah", width=300)

            # Melakukan Klasifikasi dan menampilkan hasil hanya sekali
            with st.spinner("Memproses gambar..."):
                predicted_class_label, confidence = predict_image(uploaded_file)
                time.sleep(1)  # Simulasi delay untuk menunjukkan loading
            st.success("Klasifikasi selesai!")

            # Menampilkan Klasifikasi yang telah dilakukan
            st.write(f"**Nama Motif Batik**: {predicted_class_label}")
            st.write(f"**Kepercayaan**: {confidence:.2f}%")

            # Menampilkan informasi lebih lanjut tentang motif batik
            batik_info_details = batik_info.get(predicted_class_label, {})
            if batik_info_details:
                st.write(f"**Asal**: {batik_info_details['asal']}")
                st.write(f"**Ciri-Ciri**: {batik_info_details['ciri']}")
                st.write(f"**Sejarah**: {batik_info_details['sejarah']}")
                st.write(f"**Arti Motif**: {batik_info_details['arti']}")

            # Menyimpan gambar dan hasil Klasifikasi ke riwayat
            img_bytes = io.BytesIO()
            img.save(img_bytes, format="PNG")
            img_bytes = img_bytes.getvalue()
            st.session_state.history.append(
                {
                    "image": img_bytes,
                    "label": predicted_class_label,
                    "confidence": confidence,
                }
            )


# Halaman Riwayat Klasifikasi
def show_history_page():
    if len(st.session_state.history) == 0:
        st.write("Tidak ada riwayat klasifikasi.")
    else:
        st.write("Riwayat Klasifikasi Motif Batik:")

        for i, entry in enumerate(st.session_state.history):
            # Menampilkan gambar hasil klasifikasi dengan ukuran 300x300
            img = Image.open(
                io.BytesIO(entry["image"])
            )  # Mengonversi bytes menjadi gambar
            img = img.resize((300, 300))  # Mengubah ukuran gambar menjadi 300x300

            # Menggunakan st.columns untuk menempatkan gambar di tengah
            col1, col2, col3 = st.columns(
                [1, 5, 1]
            )  # Membuat tiga kolom, kolom tengah lebih lebar
            with col2:
                st.image(
                    img,
                    caption=f"Klasifikasi {i+1}: {entry['label']} (Kepercayaan: {entry['confidence']:.2f}%)",
                    width=300,
                )

            # Menampilkan informasi hasil klasifikasi
            st.write(f"**Klasifikasi**: {entry['label']}")
            st.write(f"**Kepercayaan**: {entry['confidence']:.2f}%")

            # Menampilkan informasi lebih lanjut tentang motif batik
            predicted_class_label = entry["label"]
            batik_info_details = batik_info.get(predicted_class_label, {})
            if batik_info_details:
                st.write(f"**Asal**: {batik_info_details['asal']}")
                st.write(f"**Ciri-Ciri**: {batik_info_details['ciri']}")
                st.write(f"**Sejarah**: {batik_info_details['sejarah']}")
                st.write(f"**Arti Motif**: {batik_info_details['arti']}")

            # Menampilkan tombol untuk menghapus riwayat klasifikasi
            if st.button(f"Hapus Klasifikasi {i+1}", key=f"hapus_{i}"):
                st.session_state.history.pop(i)
                st.rerun()  # Refresh halaman setelah penghapusan

            st.markdown("---")


# Menampilkan halaman berdasarkan pilihan menu
with st.sidebar:
    selected = option_menu(
        menu_title="Main Menu",
        options=["Beranda", "Klasifikasi Batik", "Riwayat Hasil Klasifikasi"],
        icons=["house", "camera", "clock-history"],
        menu_icon="cast",
        default_index=0,
    )
if selected == "Beranda":
    show_home_page()
elif selected == "Klasifikasi Batik":
    show_classification_page()
elif selected == "Riwayat Hasil Klasifikasi":
    show_history_page()
