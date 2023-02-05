import tensorflow as tf
from keras import layers


# Test class with lstm layers in hidden layers each having 16 units (complexity of units to be increased)
class LSTMWithPositiveAndNormedWeights(tf.keras.Model):
    def __init__(self, dim, layerSize, activation=tf.nn.relu, activatePositive=tf.nn.sigmoid):
        super().__init__()
        self.nameObj = "LSTMWithPositiveAndNormedWeights"
        self.dim = dim
        self.layerSize = layerSize
        self.listOfLSTM = [layers.LSTM( 16, activation=activation) for _ in layerSize]
        self.listOfDense = [layers.Dense(dim, activation=activatePositive)]
        self.activatePositive = activatePositive

    def call(self, inputs):
        for layer in self.listOfLSTM:
            inputs = layer(inputs)
        for layer in self.listOfDense:
            inputs = layer(inputs)
        return inputs / tf.tile(tf.expand_dims(tf.reduce_sum(inputs, axis=-1), axis=-1),
                                [1, self.dim])

    def getPortfolioWeights(self, t, x):
        output = tf.concat([t * tf.ones([tf.shape(x)[0], 1]), x], axis=-1)
        for layer in self.listOfLSTM:
            output = layer(output)
        for layer in self.listOfDense:
            output = layer(output)
        return output / tf.tile(tf.expand_dims(tf.reduce_sum(output, axis=-1), axis=-1), [1, self.dim])


