from pathlib import Path
import os
import os.path

import numpy as np
import joblib

from .autoencoders import SimpleAutoencoder


class Config:
    
    BASE_PATH = Path(os.path.dirname(__file__)) / '..' / 'configs'
    
    def __init__(self, name, emnist_ae, kmnist_ae) -> None:
        self.name = name
        self.path = self.BASE_PATH / name
        self.path_emnist = self.path / 'EMNIST'
        self.path_kmnist = self.path / 'KMNIST'
        self.path_mapping = self.path / 'mapping.gz'
        
        self.emnist = SubConfig(self.path_emnist, emnist_ae)
        self.kmnist = SubConfig(self.path_kmnist, kmnist_ae)
        self.mapping = None
        
    def save(self):
        self.path.mkdir(parents=True, exist_ok=True)
        
        self.emnist.save()
        self.kmnist.save()
        self.save_mapping()

    def load(self):
        self.emnist.load()
        self.kmnist.load()
        self.load_mapping()
            
    def load_necessary(self):
        self.kmnist.load_autoencoder()
        self.kmnist.load_clusterizer()
        self.emnist.load_characters()
        self.load_mapping()
        
    def load_mapping(self):
        if self.path_mapping.exists():
            self.mapping = joblib.load(self.path_mapping)
            
    def save_mapping(self):
        if self.mapping is not None:
            joblib.dump(self.mapping, self.path_mapping)
            

class SubConfig:
    
    def __init__(self, path, autoencoder) -> None:
        self.path = path
        self.path_autoencoder = self.path / 'autoencoder'
        self.path_clusterizer = self.path / 'clusterizer.gz'
        self.path_characters = self.path / 'characters.npy'
        self.path_counts = self.path / 'counts.npy'

        self.autoencoder = autoencoder
        self.clusterizer = None
        self.characters = None
        self.counts = None

    def save(self):
        self.path.mkdir(parents=True, exist_ok=True)
        
        if self.autoencoder is not None:
            self.autoencoder.save(self.path_autoencoder)
        if self.clusterizer is not None:
            joblib.dump(self.clusterizer, self.path_clusterizer)
        if self.characters is not None:
            np.save(str(self.path_characters), self.characters)
        if self.counts is not None:
            np.save(str(self.path_counts), self.counts)

    def load(self):
        self.load_autoencoder()
        self.load_clusterizer()
        self.load_characters()
        self.load_counts()
            
    def load_autoencoder(self):
        if self.path_autoencoder.exists():
            self.autoencoder.load(self.path_autoencoder)
            
    def load_clusterizer(self):
        if self.path_clusterizer.exists():
            self.clusterizer = joblib.load(self.path_clusterizer)
            
    def load_characters(self):
        if self.path_characters.exists():
            self.characters = np.load(str(self.path_characters))
            
    def load_counts(self):
        if self.path_counts.exists():
            self.counts = np.load(str(self.path_counts))


CONFIGS = [
    Config(
        name='clean',
        emnist_ae=SimpleAutoencoder(output_features=30),
        kmnist_ae=SimpleAutoencoder(output_features=25)
    ),
    Config(
        name='noise_only',
        emnist_ae=SimpleAutoencoder(output_features=12),
        kmnist_ae=SimpleAutoencoder(output_features=11)
    ),
    Config(
        name='min_distortions_1_char',
        emnist_ae=SimpleAutoencoder(output_features=12),
        kmnist_ae=SimpleAutoencoder(output_features=11)
    ),
    Config(
        name='mid_distortions_1_char',
        emnist_ae=SimpleAutoencoder(output_features=11),
        kmnist_ae=SimpleAutoencoder(output_features=12)
    ),
    Config(
        name='min_distortions_2_char',
        emnist_ae=SimpleAutoencoder(output_features=8),
        kmnist_ae=SimpleAutoencoder(output_features=8)
    ),
    Config(
        name='min_distortions_100_char',
        emnist_ae=SimpleAutoencoder(output_features=9),
        kmnist_ae=SimpleAutoencoder(output_features=12)
    )
]


def get_config_by_name(name):
    for config in CONFIGS:
        if config.name == name:
            return config
    assert False
