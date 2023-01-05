from .test import TestAutoencoder

EMNIST_AUTOENCODERS = [
    TestAutoencoder('noise_only_autoencoder_emnist', output_features=20),
    TestAutoencoder('clean_autoencoder_emnist', output_features=30),
]

KMNIST_AUTOENCODERS = [
    TestAutoencoder('noise_only_autoencoder_kmnist', output_features=20),
    TestAutoencoder('clean_autoencoder_kmnist', output_features=30),
]

AUTOENCODERS = EMNIST_AUTOENCODERS + KMNIST_AUTOENCODERS


def get_autoencoder_by_name(name):
    ae = [autoencoder for autoencoder in AUTOENCODERS if autoencoder.name == name][0]
    ae.load()
    return ae
