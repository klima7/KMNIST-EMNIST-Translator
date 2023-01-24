import streamlit as st

from lib.data import load_book_cached, load_emnist_cached, load_kmnist_cached
from lib.generator import generate_pages
from lib.images import zip_images


book = load_book_cached()

st.title('📄 Pages Generation')

with st.expander('Book Fragment', expanded=True):
    slider_start = st.slider("Start position", max_value=len(book)-2, value=0)
    slider_end = st.slider("End position", min_value=slider_start+1, max_value=len(book), value=30_000)

with st.expander('Dataset', expanded=True):
    dataset_name = st.selectbox('Name', ('EMNIST', 'KMNIST', 'Text'), index=1)
    slider_variants = st.slider("Number of each letter variants", min_value=1, max_value=100)

with st.expander('Distortions', expanded=True):
    slider_corruption = st.slider("Corruption probability", min_value=0.0, max_value=1.0, step=0.01, value=0.3)
    slider_rotation = st.slider("Max rotation radius", max_value=45, value=30)
    slider_scale = st.slider("Max scale factor", min_value=1.0, max_value=1.30, step=0.01, value=1.15)
    slider_noise = st.slider("Salt & Pepper noise coverage", min_value=0.0, max_value=1.0, step=0.01, value=0.05)

if st.button('🔨 Generate', type='primary'):
    text = book[slider_start:slider_end]
    dataset = load_emnist_cached() if dataset_name == 'EMNIST' else load_kmnist_cached()
    images, text_images = generate_pages(text, dataset, slider_corruption, slider_noise, slider_rotation, slider_scale, slider_variants)
    if dataset_name == 'Text':
        images = text_images
    else:
        images = [(image*255).astype('uint8') for image in images]
    st.success('Pages generated successfully!', icon="✅")

    st.header('📄 Output')

    images_zip_path = zip_images(images)
    with open(images_zip_path, "rb") as f:
        btn = st.download_button(
            label="⬇ Download",
            data=f,
            file_name="images.zip",
            mime="application/zip"
        )

    st.image(images, use_column_width=True)
