import os.path
from typing import Tuple
from pathlib import Path

import streamlit as st
import numpy as np
import cv2


BOOK_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'book.txt')

EMNIST_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'emnist')

KMNIST_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'kmnist')

MAPPINGS = {
    '\n': -2, ' ': -1, 'A': 10, 'B': 11, 'C': 12, 'D': 13, 'E': 14, 'F': 15, 'G': 16, 'H': 17,
    'I': 18, 'J': 19, 'K': 20, 'L': 21, 'M': 22, 'N': 23, 'O': 24, 'P': 25, 'Q': 26, 'R': 27,
    'S': 28, 'T': 29, 'U': 30, 'V': 31, 'W': 32, 'X': 33, 'Y': 34, 'Z': 35, 'a': 36, 'b': 37,
    'c': 12, 'd': 38, 'e': 39, 'f': 40, 'g': 41, 'h': 42, 'i': 18, 'j': 19, 'k': 20, 'l': 21,
    'm': 22, 'n': 43, 'o': 24, 'p': 25, 'q': 44, 'r': 45, 's': 28, 't': 46, 'u': 30, 'v': 31,
    'w': 32, 'x': 33, 'y': 34, 'z': 35
}


def load_kmnist() -> Tuple[np.ndarray, np.ndarray]:
    train_imgs = "k49-train-imgs.npz"
    train_labels = "k49-train-labels.npz"

    images = np.load(str(Path(KMNIST_PATH) / train_imgs))["arr_0"]
    labels = np.load(str(Path(KMNIST_PATH) / train_labels))["arr_0"]

    # removing iteration character
    where = np.where(labels != 15)
    images = images[where]
    labels = labels[where]

    for i in range(len(labels)):
        if labels[i] > 15:
            labels[i] -= 1

    return _postprocess(images), labels


def load_emnist() -> Tuple[np.ndarray, np.ndarray]:
    train_imgs = "emnist-bymerge-train-images.npy"
    train_labels = "emnist-bymerge-train-labels.npy"

    images = np.load(str(Path(EMNIST_PATH) / train_imgs))
    labels = np.load(str(Path(EMNIST_PATH) / train_labels))

    return _postprocess(images), labels


def load_book() -> str:
    with open(BOOK_PATH, "r", encoding="utf8") as f:
        full_text = f.read()
        full_text = "".join(e for e in full_text if e.isalpha() or e == " " or e == "\n")
        return full_text
    
    
@st.cache
def load_book_cached():
    return load_book()


@st.cache
def load_emnist_cached():
    return load_emnist()


@st.cache
def load_kmnist_cached():
    return load_emnist()


def _postprocess(images):
    scaled_images = np.array([cv2.resize(image, (32, 32)) for image in images])
    binarized_images = (scaled_images > np.max(scaled_images) / 2) * 1.0
    return binarized_images
