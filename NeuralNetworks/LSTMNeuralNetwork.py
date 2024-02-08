import tensorflow as tf
from keras import layers

tf.config.run_functions_eagerly(True)


# Test class with lstm layers in hidden layers each having 16 units (complexity of units to be increased)
class LSTMWithPositiveAndNormedWeights(tf.keras.Model):
    def __init__(self, dim, layerSize, nbNeurons, activation=tf.nn.relu, activatePositive=tf.nn.sigmoid):
        super().__init__()
        self.nameObj = "LSTMWithPositiveAndNormedWeights"
        self.dim = dim
        self.layerSize = layerSize
        self.listOfLSTM = [layers.LSTM(nbNeurons, input_shape=(None,), activation=activation, return_sequences=True) for
                           _ in layerSize]
        self.listOfDense = [layers.Dense(dim, activation=activatePositive)]
        self.activatePositive = activatePositive

    def call(self, inputs):
        for layer in self.listOfLSTM:
            inputs = layer(inputs)
        for layer in self.listOfDense:
            inputs = layer(inputs)
        return inputs / tf.tile(tf.expand_dims(tf.reduce_sum(inputs, axis=-1), axis=-1),
                                [1, self.dim])

    def getPortfolioWeights(self, x):
        output = tf.concat([tf.ones([tf.shape(x)[0], 1]), x], axis=-1)
        # print(tf.expand_dims(output, axis=1).shape.as_list() , ' expand dim')
        output = tf.expand_dims(output, axis=1).shape.as_list()
        print(output, ' Output after shapes in NeuralNetworkClass')
        print(tf.shape(x)[0].numpy(), ' Batch size of x AS NUMPY')
        for layer in self.listOfLSTM:
            output = layer(output)
        for layer in self.listOfDense:
            output = layer(output)
        return output / tf.tile(tf.expand_dims(tf.reduce_sum(output, axis=-1), axis=-1), [1, self.dim])


class LSTMNeuralNetworkCreator:
    def __init__(self, dim, layerSize, nbNeurons, activationFunction, activatePositive=tf.nn.sigmoid):
        self.dim = dim
        self.layerSize = layerSize
        self.nbNeurons = nbNeurons
        self.activation = activationFunction
        self.activatePositive = activatePositive

    def create(self):
        return LSTMWithPositiveAndNormedWeights(self.dim, self.layerSize, self.nbNeurons, self.activation,
                                                self.activatePositive)
