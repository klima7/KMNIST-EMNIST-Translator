from io import BytesIO

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from einops import rearrange

from lib.autoencoders import EMNIST_AUTOENCODERS, KMNIST_AUTOENCODERS, get_autoencoder_by_name
from lib.translation import match_clusters
from lib.utils import visualize_clusters, get_number_of_clusters_with_eigen_values, cluster, sort_characters_by_labels, visualize_sorted_characters, visualize_clusters_learned_by_autoencoder, visualize_mapping
from lib.images import zip_images, get_uploaded_images


st.title('📝 Pages Translation')

uploaded_files = st.file_uploader('Pages to decode', accept_multiple_files=True)

emnist_autoencoder_name = st.selectbox(
    'Autoencoder EMNIST',
    [autoencoder.name for autoencoder in EMNIST_AUTOENCODERS]
)

kmnist_autoencoder_name = st.selectbox(
    'Autoencoder KMNIST',
    [autoencoder.name for autoencoder in KMNIST_AUTOENCODERS]
)

if st.button('🔨 Translate', type='primary'):
    kmnist_pages = get_uploaded_images(uploaded_files)
    kmnist_pages = 1 - kmnist_pages
    kmnist_chars = rearrange(kmnist_pages, 'p (H h) (W w) -> (p H W) h w', h=32, w=32)

    emnist_autoencoder = get_autoencoder_by_name(emnist_autoencoder_name)
    kmnist_autoencoder = get_autoencoder_by_name(kmnist_autoencoder_name)

    is_white_character = np.mean(kmnist_chars, axis=(1, 2)) < 0.1
    encoded_kmnist_chars = kmnist_autoencoder.encoder.predict(kmnist_chars)
    fil_encoded_kmnist_chars = encoded_kmnist_chars[~is_white_character]

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    visualize_clusters(fil_encoded_kmnist_chars, ax=ax, n_samples=20_000)
    st.subheader('UMAP visualization of uploaded pages')
    st.write(fig)

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    kmnist_clusters_no = get_number_of_clusters_with_eigen_values(fil_encoded_kmnist_chars, ax=ax, n_samples=3_000)
    st.subheader('Finding number of clusters')
    st.write(fig)
    st.write(f'Found {kmnist_clusters_no} clusters')

    kmnist_labels = cluster(fil_encoded_kmnist_chars, kmnist_clusters_no)
    sorted_encoded_kmnist = sort_characters_by_labels(fil_encoded_kmnist_chars, kmnist_labels)
    fig = visualize_sorted_characters(sorted_encoded_kmnist, kmnist_autoencoder, binarize=True)
    st.subheader('Found clusters')
    st.write(fig)
    
    st.subheader('Learned clusters')
    fig = visualize_clusters_learned_by_autoencoder(emnist_autoencoder)
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    st.image(buf)

    kmnist_clusters_sizes = [len(cluster) for cluster in sorted_encoded_kmnist]
    emnist_clusters_sizes = emnist_autoencoder.quantities
    mapping = match_clusters(kmnist_clusters_sizes, emnist_clusters_sizes)
    st.subheader('Clusters mapping')
    fig = visualize_mapping(mapping, sorted_encoded_kmnist, emnist_autoencoder, kmnist_autoencoder)
    st.pyplot(fig)

    translated_chars = np.zeros_like(kmnist_chars)
    emnist_labels = np.array([mapping[kmnist_label] for kmnist_label in kmnist_labels])
    translated_chars[~is_white_character] = emnist_autoencoder.characters[emnist_labels]
    translated_pages = rearrange(translated_chars, '(p H W) h w -> p (H h) (W w)', W=80, H=114, h=32, w=32)

    translated_pages = 1 - translated_pages
    translated_pages = [page for page in translated_pages]

    st.subheader('Translated pages')
    images_zip_path = zip_images(translated_pages)
    with open(images_zip_path, "rb") as f:
        btn = st.download_button(
            label="⬇ Download",
            data=f,
            file_name="translated.zip",
            mime="application/zip"
        )
    st.image(translated_pages)
