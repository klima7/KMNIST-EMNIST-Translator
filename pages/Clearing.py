import streamlit as st

from lib.images import get_uploaded_images, zip_images
from lib.autoencoders import AUTOENCODERS, get_autoencoder_by_name
from lib.clearing import clear_pages


st.title('📝 Pages Clearing')

uploaded_files = st.file_uploader('Pages to clear', accept_multiple_files=True)

autoencoder_name = st.selectbox(
    'Autoencoder',
    [autoencoder.name for autoencoder in AUTOENCODERS]
)

if st.button('🔨 Clear', type='primary'):
    dirty_pages = get_uploaded_images(uploaded_files)
    autoencoder = get_autoencoder_by_name(autoencoder_name)
    clear_pages = clear_pages(dirty_pages, autoencoder)

    st.success('Pages cleared successfully!', icon="✅")
    st.header('📄 Output')

    images_zip_path = zip_images(clear_pages)
    with open(images_zip_path, "rb") as f:
        btn = st.download_button(
            label="⬇ Download",
            data=f,
            file_name="images.zip",
            mime="application/zip"
        )

    st.image(clear_pages, use_column_width=True)
