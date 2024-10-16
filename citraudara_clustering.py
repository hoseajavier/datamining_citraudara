import streamlit as st
import numpy as np
import cv2
from PIL import Image
from sklearn.metrics import silhouette_score

# Judul Aplikasi
st.title("Image Processing and Clustering")

# 1. Upload file menggunakan widget Streamlit
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Baca citra yang diunggah
    image = np.array(Image.open(uploaded_file))

    # Tampilkan ukuran gambar asli
    st.write(f"Ukuran gambar asli: {image.shape}")  # Menampilkan seluruh dimensi gambar (height, width, channels)

    # Tangani gambar dengan 4 channel (RGBA) dengan mengonversi ke 3 channel (RGB)
    if image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        st.write(f"Channel alpha terdeteksi, gambar dikonversi ke RGB. Ukuran baru: {image.shape}")

    # Jika gambar lebih kecil dari 128x128, tambahkan padding
    if image.shape[0] < 128 or image.shape[1] < 128:
        top = (128 - image.shape[0]) // 2
        bottom = 128 - image.shape[0] - top
        left = (128 - image.shape[1]) // 2
        right = 128 - image.shape[1] - left

        # Padding untuk menambahkan border dengan warna hitam (0)
        image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        st.write(f"Gambar terlalu kecil, telah ditambahkan padding. Ukuran baru: {image.shape}")

    # Tangani gambar grayscale dengan mengonversi ke RGB
    if len(image.shape) == 2:  # Jika gambar hanya memiliki satu channel (grayscale)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)  # Konversi ke RGB

    # 2. Resize citra (misalnya, 128x128 piksel)
    resized_image = cv2.resize(image, (128, 128))

    # Tampilkan ukuran gambar setelah penanganan grayscale dan resize
    st.write(f"Ukuran gambar setelah resize: {resized_image.shape}")

    # Tampilkan gambar setelah resize
    st.image(resized_image, caption="Gambar Setelah Resize", use_column_width=True)

    # 3. Normalisasi (mengubah piksel menjadi rentang 0-1)
    normalized_image = resized_image / 255.0

    # Debugging informasi ukuran gambar setelah normalisasi
    st.write(f"Ukuran gambar setelah normalisasi: {normalized_image.shape}")

    # Tampilkan gambar setelah normalisasi
    st.image(normalized_image, caption="Gambar Setelah Normalisasi", use_column_width=True)

    # 4. Flatten gambar untuk keperluan clustering
    try:
        pixels = normalized_image.reshape(-1, 3)  # Flatten menjadi bentuk (128*128, 3)
        st.write(f"Ukuran array setelah di-flatten: {pixels.shape}")  # Seharusnya menghasilkan (16384, 3)
    except ValueError as e:
        st.error(f"Error dalam proses reshape: {e}")
        st.stop()  # Hentikan eksekusi jika ada error

    # Fungsi K-Means++ untuk inisialisasi yang lebih baik
    def initialize_centroids(pixels, k):
        centroids = [pixels[np.random.choice(len(pixels))]]
        for _ in range(1, k):
            distances = np.min([np.linalg.norm(pixels - centroid, axis=1) for centroid in centroids], axis=0)
            probabilities = distances / distances.sum()
            cumulative_probabilities = np.cumsum(probabilities)
            r = np.random.rand()
            next_centroid = pixels[np.searchsorted(cumulative_probabilities, r)]
            centroids.append(next_centroid)
        return np.array(centroids)

    # Fungsi clustering manual menggunakan k-means++ dengan centroid initialization
    def kmeans_manual(pixels, k, max_iter=100):
        np.random.seed(42)
        centroids = initialize_centroids(pixels, k)
        for _ in range(max_iter):
            distances = np.linalg.norm(pixels[:, np.newaxis] - centroids, axis=2)
            labels = np.argmin(distances, axis=1)
            new_centroids = np.array([pixels[labels == i].mean(axis=0) for i in range(k)])
            if np.all(centroids == new_centroids):
                break
            centroids = new_centroids
        return labels, centroids

    # 5. Slider untuk memilih jumlah cluster
    k = st.slider("Select number of clusters", min_value=2, max_value=6, value=3)

    # Clustering dengan k-means
    if st.button(f"Apply Clustering with {k} Clusters"):
        labels, centroids = kmeans_manual(pixels, k)

        # Mengubah label menjadi warna centroid
        clustered_image = centroids[labels].reshape(128, 128, 3)

        # Konversi gambar kembali ke rentang 0-255 (karena saat ini masih dalam rentang 0-1)
        clustered_image = (clustered_image * 255).astype(np.uint8)

        # Visualisasi hasil clustering
        st.image(clustered_image, caption=f"Clustering dengan {k} Cluster", use_column_width=True)

        # 6. Hitung dan tampilkan Silhouette Coefficient
        silhouette_avg = silhouette_score(pixels, labels)
        st.write(f"Silhouette Coefficient untuk {k} cluster: {silhouette_avg:.3f}")

    # 7. Deteksi tepi menggunakan operator Sobel
    gray_image = cv2.cvtColor(resized_image, cv2.COLOR_RGB2GRAY)  # Konversi ke grayscale untuk deteksi tepi
    grad_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)  # Gradien arah horizontal
    grad_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)  # Gradien arah vertikal
    magnitude = cv2.magnitude(grad_x, grad_y)  # Magnitudo gradien

    # Tampilkan hasil deteksi tepi
    st.subheader("Deteksi Tepi Menggunakan Sobel")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.image(grad_x, caption="Gradien Horizontal", use_column_width=True, clamp=True, channels="GRAY")

    with col2:
        st.image(grad_y, caption="Gradien Vertikal", use_column_width=True, clamp=True, channels="GRAY")

    with col3:
        st.image(magnitude, caption="Magnitudo Gradien", use_column_width=True, clamp=True, channels="GRAY")
