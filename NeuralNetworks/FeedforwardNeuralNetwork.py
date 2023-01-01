import tensorflow as tf
from keras import layers

# Give back a weight between 0 and 1 with sum equal to one
# last activation function is sigmoid, yielding values in [0,1] corresponding to weights
############################################################################

class FeedForwardWithPositiveAndNormedWeights(tf.keras.Model):
    def __init__(self, dim, layerSize, activation=tf.nn.relu, activatePositive=tf.nn.sigmoid):
        super().__init__()
        self.nameObj = "FeedForwardWithPositiveAndNormedWeights"
        self.dim = dim
        self.layerSize = layerSize
        self.listOfDense = [layers.Dense(nbNeurons, activation=activation) for nbNeurons in layerSize]
        self.listOfDense.append(layers.Dense(dim, activation=activatePositive))
        self.activatePositive = activatePositive

    def call(self, inputs):
        for layer in self.listOfDense:
            inputs = layer(inputs)
        return inputs / tf.tile(tf.expand_dims(tf.reduce_sum(inputs, axis=-1), axis=-1),
                                [1, self.dim])

    def getPortfolioWeights(self, t, x):
        output = tf.concat([t * tf.ones([tf.shape(x)[0], 1]), x], axis=-1)
        for layer in self.listOfDense:
            output = layer(output)
        return output / tf.tile(tf.expand_dims(tf.reduce_sum(output, axis=-1), axis=-1), [1, self.dim])


class FeedForwardWithPositiveAndNormedWeightsCreator:
    def __init__(self, dim, layerSize, activationFunction, activatePositive=tf.nn.sigmoid):
        self.dim = dim
        self.layerSize = layerSize
        self.activation = activationFunction
        self.activatePositive = activatePositive

    def create(self):
        return FeedForwardWithPositiveAndNormedWeights(self.dim, self.layerSize, self.activation, self.activatePositive)
