import numpy as np
from keras import Input
from keras.layers import Activation, BatchNormalization, Conv2D, Dense, Flatten, Conv2DTranspose, Reshape
from keras.models import Model, Sequential

from .base import BaseAutoencoder


class TestAutoencoder(BaseAutoencoder):

    def __init__(self, path: str, output_features: int = 16) -> None:
        self.output_features = output_features
        super().__init__(path)

    def _create_encoder_and_decoder(self):
        encoder_conv_part = self.__encoder_conv_part()

        encoder = Sequential(
            [
                encoder_conv_part,
                Flatten(),
                Dense(np.prod(encoder_conv_part.output_shape[1:]), activation="ReLU"),
                Dense(self.output_features, activation='tanh'),
            ]
        )

        decoder = Sequential(
            [
                Input(shape=(self.output_features,)),
                Dense(np.prod(encoder_conv_part.output_shape[1:]), activation="ReLU"),
                Reshape(encoder_conv_part.output_shape[1:]),
                self.__decoder_conv_part(),
            ]
        )

        return encoder, decoder

    def __encoder_conv_part(self) -> Model:
        return Sequential(
            [
                Input(self.INPUT_SHAPE),
                self.__encoder_block(16),
                self.__encoder_block(32),
                self.__encoder_block(64),
            ]
        )

    def __decoder_conv_part(self) -> Model:
        return Sequential(
            [
                self.__decoder_block(64),
                self.__decoder_block(32),
                self.__decoder_block(16),
                Conv2D(1, 3, activation='sigmoid', padding='same'),
            ]
        )

    @staticmethod
    def __encoder_block(filters: int) -> Model:
        return Sequential(
            [
                Conv2D(filters, kernel_size=3, strides=2, padding='same'),
                BatchNormalization(),
                Activation('LeakyReLU'),
            ]
        )

    @staticmethod
    def __decoder_block(filters: int) -> Model:
        return Sequential(
            [
                Conv2DTranspose(filters, 3, strides=2, padding='same'),
                BatchNormalization(),
                Activation('LeakyReLU'),
            ]
        )
