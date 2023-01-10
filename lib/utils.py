import random
import warnings

import matplotlib.pyplot as plt
import numpy as np
import cv2
import umap
from umap import plot as uplot
from sklearn.cluster import KMeans, SpectralClustering
from yellowbrick.cluster import KElbowVisualizer
from scipy.linalg import eigh
from scipy import sparse
from sklearn.metrics import pairwise_kernels


def sample(data, n_samples):
    if n_samples is None or n_samples > len(data):
        return data
    indexes = np.arange(len(data))
    np.random.shuffle(indexes)
    indexes = indexes[:n_samples]
    sampled_data = data[indexes]
    return sampled_data


def visualize_clusters(encoded_characters, ax, n_samples=None):
    encoded_characters = sample(encoded_characters, n_samples)
    with warnings.catch_warnings():
        embedding = umap.UMAP(n_neighbors=5, n_components=2).fit(encoded_characters)
        uplot.points(embedding, width=1200, height=1200, ax=ax, show_legend=False, labels=np.zeros((len(encoded_characters),)))


def get_number_of_clusters_with_elbow_method(encoded_characters, ax, n_samples=None):
    encoded_characters = sample(encoded_characters, n_samples)
    
    # model = KMeans(n_init='auto')
    model = SpectralClustering()
    visualizer = KElbowVisualizer(model, k=(15, 50), timings=False, ax=ax)
    visualizer.fit(encoded_characters)
    return visualizer.elbow_value_


def get_number_of_clusters_with_eigen_values(encoded_characters, ax, n_samples=None):
    encoded_characters = sample(encoded_characters, n_samples)

    adjacency = pairwise_kernels(encoded_characters, metric='rbf', filter_params=True, gamma=2, degree=3, coef0=1)
    laplacian, _ = sparse.csgraph.laplacian(adjacency, normed=True, return_diag=True)
    lambdas, _ = eigh(laplacian)
    lambdas = np.sort(lambdas)[:50]
    ax.plot(lambdas)
    
    diffs = lambdas[1:] - lambdas[:-1]
    elbow = np.argmax(diffs)
    ax.axvline(x=elbow, color='black', linestyle='--')
    return elbow


def cluster(encoded_characters, clusters_no):
    model = SpectralClustering(n_clusters=clusters_no)
    labels = model.fit_predict(encoded_characters)
    labels = rewrite_labels(labels)
    return labels


def sort_characters_by_labels(characters, labels):
    sorted_characters = []

    min_label = np.min(labels)
    max_label = np.max(labels)

    for label in range(min_label, max_label+1):
        label_characters = characters[labels == label]
        sorted_characters.append(label_characters)
        
    return sorted_characters


def visualize_sorted_characters(sorted_encoded_chars, autoencoder, binarize=True):
    total = sum([len(a) for a in sorted_encoded_chars])
    rows = []

    for i, class_encoded_characters in enumerate(sorted_encoded_chars):
        count = len(class_encoded_characters)
        percent = count / total * 100
        text = np.zeros((32, 90))
        cv2.putText(text, f'{i}', org=(0, 26), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, color=(255, 255, 255))
        cv2.putText(text, f'{count}', org=(35, 16), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(255, 255, 255))
        cv2.putText(text, f'{percent:.2f}%', org=(35, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(255, 255, 255))
        
        random_encoded_characters = np.array(random.choices(class_encoded_characters, k=12))
        random_decoded_characters = np.squeeze(autoencoder.decoder.predict(random_encoded_characters, verbose=0))
        
        characters_row = np.hstack([text, *random_decoded_characters])
        rows.append(characters_row)

    image = np.vstack(rows)
    if binarize:
        image = (image > 0.5) * 1.0
    fig, ax = plt.subplots(1, 1, figsize=(12, len(sorted_encoded_chars)*2))
    plt.tick_params(left = False, right = False , labelleft = False , labelbottom = False, bottom = False)
    ax.imshow(image)
    ax.grid(None)
    return fig


def visualize_clusters_learned_by_autoencoder(ae):
    rows = []
    total = np.sum(ae.quantities)
    
    for i, (char, count) in enumerate(zip(ae.characters, ae.quantities)):
        percent = count / total * 100
        
        text = np.zeros((32, 90))
        cv2.putText(text, f'{i}', org=(0, 26), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, color=(255, 255, 255))
        cv2.putText(text, f'{count}', org=(35, 16), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(255, 255, 255))
        cv2.putText(text, f'{percent:.2f}%', org=(35, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(255, 255, 255))
        
        row = np.hstack([text/255, char])
        rows.append(row)
        
    image = np.vstack(rows)
    fig, ax = plt.subplots(1, 1, figsize=(3, len(ae.quantities)*1))
    plt.tick_params(left = False, right = False , labelleft = False , labelbottom = False, bottom = False)
    ax.imshow(image)
    ax.grid(None)
    return fig


def visualize_mapping(mapping, sorted_encoded_kmnist, emnist_ae, kmnist_ae):
    separator = np.zeros((32, 32))
    cv2.line(separator, (5, 16), (27, 16), (255, 255, 255), 2)
    cv2.line(separator, (27, 16), (27-6, 16-6), (255, 255, 255), 2)
    cv2.line(separator, (27, 16), (27-6, 16+6), (255, 255, 255), 2)
    separator /= 255
    
    e_total = sum(emnist_ae.quantities)
    k_total = sum([len(x) for x in sorted_encoded_kmnist])

    rows = []
    for from_label, to_label in mapping.items():
        e_count = emnist_ae.quantities[to_label]
        k_count = len(sorted_encoded_kmnist[from_label])
        
        e_text = create_count_image(to_label, e_count, e_total)
        k_text = create_count_image(from_label, k_count, k_total)
        
        k_chars_encoded = np.array(random.choices(sorted_encoded_kmnist[from_label], k=4))
        k_chars = np.squeeze(kmnist_ae.decoder.predict(k_chars_encoded))
        e_char = emnist_ae.characters[to_label]
        
        print(e_char.shape)
        print(k_chars.shape)
        
        row = np.hstack([k_text, *k_chars, separator, e_text, e_char])
        rows.append(row)
        
    image = np.vstack(rows)
    fig, ax = plt.subplots(1, 1, figsize=(8, len(mapping)*1))
    plt.tick_params(left = False, right = False , labelleft = False , labelbottom = False, bottom = False)
    ax.imshow(image)
    ax.grid(None)
    return fig


def create_count_image(no, count, total):
    percent = count / total * 100
    image = np.zeros((32, 90))
    cv2.putText(image, f'{no:2}', org=(0, 26), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, color=(255, 255, 255))
    cv2.putText(image, f'{count}', org=(35, 16), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(255, 255, 255))
    cv2.putText(image, f'{percent:.2f}%', org=(35, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(255, 255, 255))
    image /= 255
    return image


def rewrite_labels(labels):
    old_labels, counts = np.unique(labels, return_counts=True)
    indexes = np.flip(np.argsort(counts))
    old_labels = old_labels[indexes]
    new_labels = np.zeros_like(labels)
    for new_label, old_label in enumerate(old_labels):
        new_labels[labels == old_label] = new_label
    return new_labels
