import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from einops import rearrange
import umap
from umap import plot as uplot

from lib.configs import CONFIGS, get_config_by_name
from lib.images import get_uploaded_images
from lib.utils import filter_noise, visualize_clusters, sample, sort_by_labels, show_sorted_characters
from lib.data import MAPPINGS


def show_umap(encoded_chars, labels, true_labels):
    if len(true_labels) != len(labels):
        true_labels = np.array(labels)
        true_labels_present = False
    else:
        true_labels_present = True
        
    samp_encoded_chars, samp_labels, samp_true_labels = sample(encoded_chars, labels, true_labels, n=5_000)
    embedding = umap.UMAP(n_neighbors=5, n_components=2).fit(samp_encoded_chars)
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    uplot.points(embedding, width=1200, height=1200, ax=ax, show_legend=False, labels=samp_labels)
    plt.title('UMAP (with predicted labels)')
    st.subheader('UMAP visualization')
    st.write(fig)
    
    if true_labels_present:
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        uplot.points(embedding, width=1200, height=1200, ax=ax, show_legend=False, labels=samp_true_labels)
        plt.title('UMAP (with true labels)')
        st.write(fig)
        
        
st.title('📝 Pages Translation')

config_name = st.selectbox(
    'Config',
    [config.name for config in CONFIGS]
)

kmnist_pages = st.file_uploader('Pages to decode', accept_multiple_files=True)

text_pages = st.file_uploader('Text pages (optional)', accept_multiple_files=True)

should_show_umap = st.checkbox('Show UMAP')

if st.button('🔨 Translate', type='primary'):
    with st.spinner('Fetching uploaded pages'):
        kmnist_pages, pages_names = get_uploaded_images(kmnist_pages, with_names=True)
        kmnist_pages = kmnist_pages.astype(np.float64) / 255
    
    with st.spinner('Loading config'):
        config = get_config_by_name(config_name)
        config.load_necessary()
        econf = config.emnist
        kconf = config.kmnist
    
    with st.spinner('Filtering noise'):
        example_original = kmnist_pages[0, :256, :256]
        kmnist_pages = filter_noise(kmnist_pages)
        example_filtered = kmnist_pages[0, :256, :256]
        filter_example = np.hstack([example_original, example_filtered])
        st.subheader('Filtering noise')
        st.image(filter_example)
    
    kmnist_pages = 1 - kmnist_pages
    all_chars = rearrange(kmnist_pages, 'p (H h) (W w) -> (p H W) h w', h=32, w=32)

    text_pages = get_uploaded_images(text_pages)
    true_labels = np.array([MAPPINGS[chr(c)] for c in text_pages.flatten()])
    structured_text = [[''.join([chr(c) for c in line]) for line in page] for page in text_pages]

    with st.spinner('Filtering white characters'):
        is_white_character = np.mean(all_chars, axis=(1, 2)) < 0.05
        if len(true_labels) == len(all_chars):
            true_labels = true_labels[~is_white_character]
        chars = all_chars[~is_white_character]
    
    with st.spinner('Running AE and clusterizer'):
        encoded_chars = kconf.autoencoder.encoder.predict(chars)
        labels = kconf.clusterizer.predict(encoded_chars)

    if should_show_umap:
        with st.spinner('Calculating UMAP'):
            show_umap(encoded_chars, labels, true_labels)
            
            st.subheader('Characters clustered together')
            sorted_encoded_chars = sort_by_labels(encoded_chars, labels)
            fig = show_sorted_characters(sorted_encoded_chars, kconf.autoencoder, binarize=True)
            st.write(fig)

    with st.spinner('Swapping characters'):
        trans_chars = np.zeros_like(all_chars)
        trans_labels = np.array([config.mapping[label] for label in labels])
        trans_chars[~is_white_character] = econf.characters[trans_labels]
        trans_pages = rearrange(trans_chars, '(p H W) h w -> p (H h) (W w)', W=80, H=114, h=32, w=32)
        trans_pages = 1 - trans_pages
        trans_pages = [page for page in trans_pages]
    
    for i in range(len(trans_pages)):
        name = pages_names[i]
        trans_page = trans_pages[i]
        oryginal_page = kmnist_pages[i]
        st.subheader(f'Page {name}')
        
        st.image(trans_page)
        
        if len(true_labels) == len(labels):
            page_text = structured_text[i]
            with st.expander('Book text', expanded=False):
                st.text('\n'.join(page_text))
                
        with st.expander('Original page', expanded=False):
            st.image(1-oryginal_page)
