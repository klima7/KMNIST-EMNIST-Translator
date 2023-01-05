from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
from keras.models import Model, Sequential


class BaseAutoencoder(ABC):

    INPUT_SHAPE = (32, 32, 1)

    def __init__(self, name):
        self.name = name
        self.path = Path('configs') / name
        self.quantities = np.array([])
        self.characters = np.array([])

        self.encoder, self.decoder = self._create_encoder_and_decoder()
        self.model = Sequential([self.encoder, self.decoder])

        self.__compile()

    def __compile(self):
        self.encoder.compile()
        self.decoder.compile()
        self.model.compile(optimizer='adam', loss='mse')

    @abstractmethod
    def _create_encoder_and_decoder(self) -> Model:
        pass

    def save(self):
        self.path.mkdir(parents=True, exist_ok=True)
        self.encoder.save(str(self.path / 'encoder.h5'))
        self.decoder.save(str(self.path / 'decoder.h5'))
        self.model.save(str(self.path / 'model.h5'))
        np.save(str(self.path / 'quantities.npy'), self.quantities)
        np.save(str(self.path / 'characters.npy'), self.characters)

    def load(self):
        self.encoder(np.zeros((1, *self.INPUT_SHAPE)))
        self.decoder(np.zeros((1, *self.encoder.output_shape[1:])))
        self.model(np.zeros((1, *self.INPUT_SHAPE)))

        self.encoder.load_weights(self.path / 'encoder.h5')
        self.decoder.load_weights(self.path / 'decoder.h5')
        self.model.load_weights(self.path / 'model.h5')
        self.quantities = np.load(str(self.path / 'quantities.npy'))
        self.characters = np.load(str(self.path / 'characters.npy'))
