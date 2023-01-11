import random

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

from .clusterizer import EnhancedSpectralClustering


def sample(*data, n):
    length = len(data[0])
    all_lengths = np.array([len(elem) for elem in data])
    assert np.all(all_lengths == length)
    
    if n is None or n > length:
        return data
    
    indexes = np.arange(length)
    np.random.shuffle(indexes)
    indexes = indexes[:n]
    sampled_data = [data_elem[indexes] for data_elem in data]
    if len(sampled_data) == 1:
        return sampled_data[0]
    return sampled_data


def visualize_clusters(encoded_characters, ax=None, labels=None, n_samples=None):
    if labels is None:
        labels = np.zeros((len(encoded_characters),))
        
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(10, 10))
        
    encoded_characters, labels = sample(encoded_characters, labels, n=n_samples)
    embedding = umap.UMAP(n_neighbors=5, n_components=2).fit(encoded_characters)
    uplot.points(embedding, width=1200, height=1200, ax=ax, show_legend=False, labels=labels)
    plt.title('UMAP visualization')


def get_number_of_clusters_with_eigen_values(encoded_characters, n_samples=None):
    encoded_characters = sample(encoded_characters, n=n_samples)

    adjacency = pairwise_kernels(encoded_characters, metric='rbf', filter_params=True, gamma=1, degree=3, coef0=1)
    laplacian, _ = sparse.csgraph.laplacian(adjacency, normed=True, return_diag=True)
    lambdas, _ = eigh(laplacian)
    lambdas = np.sort(lambdas)[:50]
    
    _, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.plot(lambdas)
    diffs = lambdas[1:] - lambdas[:-1]
    elbow = np.argmax(diffs)+1
    ax.axvline(x=elbow, color='black', linestyle='--')
    ax.annotate(str(elbow), (elbow+0.5, 0), fontsize=20)
    ax.set_title('Eigenvalues used to find number of clusters')
    ax.set_xlabel('Index of eigenvalue')
    ax.set_ylabel('Eigenvalue value')
    return elbow


def sort_by_labels(data, labels):
    sorted_data = []

    for label in np.sort(np.unique(labels)):
        sorted_data.append(data[labels == label])
        
    return sorted_data


def show_sorted_characters(sorted_encoded_chars, autoencoder, binarize=True):
    total = sum([len(a) for a in sorted_encoded_chars])
    sorted_encoded_chars = sorted(enumerate(sorted_encoded_chars), key=lambda x: len(x[1]), reverse=True)
    
    rows = []

    for i, class_encoded_characters in sorted_encoded_chars:
        text = create_count_image(i, len(class_encoded_characters), total)
        
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
    ax.set_title('Sorted characters from found clusters')
    return fig


def show_clusters_consistency_matrix(sorted_labels):
    unique_labels = np.unique(np.concatenate(sorted_labels))
    unique_labels = unique_labels[unique_labels >= 0]
    
    ordered_zip = sorted(enumerate(sorted_labels), key=lambda x: len(x[1]), reverse=True)
    ordered_indexes = [entry[0] for entry in ordered_zip]
    ordered_labels = [entry[1] for entry in ordered_zip]
    
    all_counts = []
    
    for label in unique_labels:
        counts = np.array([np.sum(labels == label) for labels in ordered_labels])
        all_counts.append(counts)
        
    counts_matrix = np.array(all_counts)
    percents_matrix = counts_matrix / np.sum(counts_matrix, axis=0)
    bin_percents_matrix = percents_matrix > 0.2
    
    real_counts = np.sum(bin_percents_matrix, axis=1)
    pred_counts = np.sum(bin_percents_matrix, axis=0)
        
    fig, ax = plt.subplots(1, 1, figsize=(9, 9))
    ax.matshow(percents_matrix)
    ax.set_xticks(np.arange(len(sorted_labels)))
    ax.set_yticks(np.arange(len(unique_labels)))
    ax.set_xticklabels([f'{x}\n({count})' for x, count in zip(ordered_indexes, pred_counts)])
    ax.set_yticklabels([f'{y} ({count})' for y, count in zip(unique_labels, real_counts)])
    ax.tick_params(axis="x", bottom=True, top=True, labelbottom=True, labeltop=True)
    ax.tick_params(axis="y", left=True, right=True, labelleft=True, labelright=True)
    ax.set_ylabel('Real cluster')
    ax.set_xlabel('Assigned cluster (sorted by size)')


def create_count_image(no, count, total):
    percent = count / total * 100
    image = np.zeros((32, 90))
    cv2.putText(image, f'{no:2}', org=(0, 26), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, color=(255, 255, 255))
    cv2.putText(image, f'{count}', org=(35, 16), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(255, 255, 255))
    cv2.putText(image, f'{percent:.2f}%', org=(35, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(255, 255, 255))
    image /= 255
    return image


def create_arrow_image():
    arrow = np.zeros((32, 32))
    cv2.line(arrow, (5, 16), (27, 16), (255, 255, 255), 2)
    cv2.line(arrow, (27, 16), (27-6, 16-6), (255, 255, 255), 2)
    cv2.line(arrow, (27, 16), (27-6, 16+6), (255, 255, 255), 2)
    return arrow / 255


def rewrite_labels(labels):
    old_labels, counts = np.unique(labels, return_counts=True)
    indexes = np.flip(np.argsort(counts))
    old_labels = old_labels[indexes]
    new_labels = np.zeros_like(labels)
    for new_label, old_label in enumerate(old_labels):
        new_labels[labels == old_label] = new_label
    return new_labels


def show_sorted_bar_chart(heights, labels=None, title=None):
    if labels is None:
        labels = range(len(heights))
    sorted_zip = sorted(zip(heights, labels), reverse=True)
    sorted_heights = [elem[0] for elem in sorted_zip]
    sorted_labels = [elem[1] for elem in sorted_zip]
    plt.bar([str(label) for label in sorted_labels], sorted_heights)
    plt.title(title)
