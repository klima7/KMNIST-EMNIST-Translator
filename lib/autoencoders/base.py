from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
from keras.models import Sequential


class BaseAutoencoder(ABC):

    INPUT_SHAPE = (32, 32, 1)

    def __init__(self):
        self.encoder, self.decoder = self._create_encoder_and_decoder()
        self.model = Sequential([self.encoder, self.decoder])
        self.__compile()

    def __compile(self):
        self.encoder.compile()
        self.decoder.compile()
        self.model.compile(optimizer='adam', loss='mse')

    @abstractmethod
    def _create_encoder_and_decoder(self):
        pass

    def save(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.encoder.save(str(path / 'encoder.h5'))
        self.decoder.save(str(path / 'decoder.h5'))
        self.model.save(str(path / 'model.h5'))

    def load(self, path):
        path = Path(path)
        
        self.encoder(np.zeros((1, *self.INPUT_SHAPE)))
        self.decoder(np.zeros((1, *self.encoder.output_shape[1:])))
        self.model(np.zeros((1, *self.INPUT_SHAPE)))

        self.encoder.load_weights(path / 'encoder.h5')
        self.decoder.load_weights(path / 'decoder.h5')
        self.model.load_weights(path / 'model.h5')
