import random
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
from einops import rearrange

from .data import MAPPINGS
from .utils import create_count_image, create_arrow_image


def load_pages_from_dir(dir_path, pages_count):
    pages = []
    for page_no in range(pages_count):
        page_name = f'{page_no}.png'
        path = str(Path(dir_path) / page_name)
        page = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        pages.append(page)
    return np.array(pages)


def load_dataset_pages(name, pages_count):
    emnist_path = Path('..') / 'datasets' / name / 'EMNIST'
    kmnist_path = Path('..') / 'datasets' / name / 'KMNIST'
    text_path = Path('..') / 'datasets' / name / 'text'

    emnist_pages = load_pages_from_dir(emnist_path, pages_count).astype(np.float64) / 255
    kmnist_pages = load_pages_from_dir(kmnist_path, pages_count).astype(np.float64) / 255
    text_pages = load_pages_from_dir(text_path, pages_count)

    return emnist_pages, kmnist_pages, text_pages


def filter_white_characters(emnist_chars, kmnist_chars, labels, threshold):
    filtered_emnist = []
    filtered_kmnist = []
    fil_labels = []

    for emnist_char, kmnist_char, label in zip(emnist_chars, kmnist_chars, labels):
        if np.average(emnist_char) < threshold and np.average(kmnist_char) < threshold:
            continue
        filtered_emnist.append(emnist_char)
        filtered_kmnist.append(kmnist_char)
        fil_labels.append(label)

    drop_percent = (len(emnist_chars) - len(filtered_emnist)) / len(emnist_chars) * 100
    removed_count = np.sum(labels!=-1) - np.sum(np.array(fil_labels)!=-1)
    keept_count = np.sum(fil_labels == -1)
    print(f'Dropped {drop_percent:.1f}% characters')
    print(f'Incorrectly removed characters: {removed_count}')
    print(f'Whitespaces missed: {keept_count}')

    return np.array(filtered_emnist), np.array(filtered_kmnist), np.array(fil_labels)


def show_datasets(emnist_chars, kmnist_chars, samples_count):
    indexes = random.choices(range(len(emnist_chars)), k=samples_count)
    selected_emnist_chars = emnist_chars[indexes]
    selected_kmnist_chars = kmnist_chars[indexes]
    image = np.vstack([np.hstack(selected_emnist_chars), np.hstack(np.squeeze(selected_kmnist_chars))])
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.imshow(image, 'gray')


def test_autoencoder(autoencoder, characters, samples_count, binarize=False):
    selected_chars = np.array(random.choices(characters, k=samples_count))
    encoded_chars = autoencoder.encoder.predict(selected_chars, verbose=0)
    decoded_chars = autoencoder.decoder.predict(encoded_chars, verbose=0)
    if binarize:
        decoded_chars = (decoded_chars > 0.5) * 1.0
    image = np.vstack([np.hstack(selected_chars), np.hstack(np.squeeze(decoded_chars))])

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.imshow(image, 'gray')


def create_characters_representatives(sorted_encoded_chars, autoencoder):
    encoded_representatives = np.array([np.mean(class_encoded_chars, axis=0)
                                        for class_encoded_chars in sorted_encoded_chars])
    representatives = np.squeeze(autoencoder.decoder.predict(encoded_representatives, verbose=0))
    representatives = (representatives > 0.5) * 1.0
    return representatives


def show_characters_representatives(representative_chars):
    representative_chars = np.array(representative_chars)
    for _ in range(7-len(representative_chars) % 7):
        representative_chars = np.vstack([representative_chars, np.zeros((1, 32, 32))])
    image = rearrange(representative_chars, '(H W) h w -> (H h) (W w)', W=7)
    _, ax = plt.subplots(1, 1, figsize=(12, 12))
    ax.imshow(1-image, cmap='gray')


def show_subconfig_characters(subconfig):
    total = np.sum(subconfig.counts)
    elems = []
    
    for i, (char, count) in enumerate(zip(subconfig.characters, subconfig.counts)):
        text = create_count_image(i, count, total)
        row = np.hstack([text, char])
        elems.append(row)
        
    for _ in range((4 - len(elems) % 4) % 4):
        elems.append(np.zeros_like(elems[0]))
        
    image = rearrange(elems, '(H W) h w -> (H h) (W w)', W=4)
    fig, ax = plt.subplots(1, 1, figsize=(10, (len(elems) / 4) * 2))
    plt.tick_params(left = False, right = False , labelleft = False , labelbottom = False, bottom = False)
    ax.imshow(1-image, cmap='gray')
    return fig


def create_mapping(count_from, counts_to):
    idxs_from = np.flip(np.argsort(count_from))
    idxs_to = np.flip(np.argsort(counts_to))
    
    mapping = {}

    for i, idx_from in enumerate(idxs_from):
        if i < len(idxs_to):
            mapping[idx_from] = idxs_to[i]
        else:
            mapping[idx_from] = idxs_to[-1]

    return mapping


def show_mapping(conf):
    econf, kconf = conf.emnist, conf.kmnist
    arrow = create_arrow_image()
    
    e_total = sum(econf.counts)
    k_total = sum(kconf.counts)

    rows = []
    for k_label in np.flip(np.argsort(kconf.counts)):
        e_label = conf.mapping[k_label]
        
        e_count = econf.counts[e_label]
        k_count = kconf.counts[k_label]
        
        e_text = create_count_image(e_label, e_count, e_total)
        k_text = create_count_image(k_label, k_count, k_total)
        
        e_char = econf.characters[e_label]
        k_char = kconf.characters[k_label]
        
        row = np.hstack([k_text, k_char, arrow, e_text, e_char])
        rows.append(row)
        
    image = np.vstack(rows)
    _, ax = plt.subplots(1, 1, figsize=(5, len(econf.counts)*1))
    plt.tick_params(left = False, right = False , labelleft = False , labelbottom = False, bottom = False)
    ax.imshow(1-image, cmap='gray')
