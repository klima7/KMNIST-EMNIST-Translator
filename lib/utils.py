import random
import warnings

import matplotlib.pyplot as plt
import numpy as np
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
    
    elbow = np.nonzero(lambdas < 0.4)[0][-1]
    ax.axvline(x=elbow, color='black', linestyle='--')
    return elbow


def cluster(encoded_characters, clusters_no):
    model = SpectralClustering(n_clusters=clusters_no)
    return model.fit_predict(encoded_characters)


def sort_characters_by_labels(characters, labels):
    sorted_characters = []

    min_label = np.min(labels)
    max_label = np.max(labels)

    for label in range(min_label, max_label+1):
        label_characters = characters[labels == label]
        sorted_characters.append(label_characters)

    return sorted_characters


def visualize_sorted_characters(sorted_encoded_chars, autoencoder, binarize=True):
    rows = []

    for class_encoded_characters in sorted_encoded_chars:
        random_encoded_characters = np.array(random.choices(class_encoded_characters, k=12))
        random_decoded_characters = np.squeeze(autoencoder.decoder.predict(random_encoded_characters, verbose=0))
        characters_row = np.hstack(random_decoded_characters)
        rows.append(characters_row)

    image = np.vstack(rows)
    if binarize:
        image = (image > 0.5) * 1.0
    fig, ax = plt.subplots(1, 1, figsize=(12, len(sorted_encoded_chars)*2))
    ax.imshow(image)
    ax.grid(None)
    return fig
