import os
import time

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from AssetModel.BlackScholesModelWithBankAccount import BlackScholesModelWithBankAccount


# Combines all objects needed (kerasModel,AssetModel,Neuralnetwork) to create a Optimizer which yields objective function
# and can be trained.

class MeanCVaROptimizer:

    def __init__(self, percentile, assetModel, kerasModel, betaList, portValue, T, dt, initialLR,
                 finalLR):
        self.assetModel = assetModel
        self.kerasModel = kerasModel
        self.T = T
        self.N = int((T + 1e-10) / dt)
        self.dt = T / self.N
        self.portValue = portValue
        self.initialLR = initialLR
        self.optimizer = tf.optimizers.Adam(learning_rate=initialLR)
        self.finalLR = finalLR
        self.percentile = percentile
        self.betaList = tf.constant(betaList, dtype=tf.float32)
        self.name = "MeanCVaROptimizer"
        self.loss = []

    # Manual decrease the learning rate
    def linearDecrease(self, iteration, numEpochs):
        self.optimizer.learning_rate.assign(
            (self.finalLR - self.initialLR) / numEpochs * iteration + self.initialLR)

    @tf.function
    def simPorfolio(self, nbSimul, beta):
        assetPrev = self.assetModel.initValues()
        assetPrev = tf.tile(tf.constant(assetPrev, shape=[1, self.assetModel.size()], dtype=tf.float32), [nbSimul, 1])

        portValueCur = self.portValue * tf.ones([nbSimul])
        StateForWeights = tf.stack([portValueCur], axis=1)

        stateForWeightsAndBeta = tf.concat([StateForWeights, beta * tf.ones([nbSimul, 1])], axis=1)
        weights = self.kerasModel.getPortfolioWeights(0, stateForWeightsAndBeta)
        for i in range(self.N):
            asset = self.assetModel.oneStep(assetPrev, self.dt)
            yieldAsset = (asset - assetPrev) / assetPrev
            portValueCur = portValueCur * (1 + tf.reduce_sum(weights * yieldAsset, axis=-1))
            if i < self.N - 1:
                tNext = (i + 1) * self.dt
                stateForWeights = tf.stack([portValueCur], axis=1)
                stateForWeightsAndBeta = tf.concat([stateForWeights, beta * tf.ones([nbSimul, 1])], axis=1)
                weights = self.kerasModel.getPortfolioWeights(tNext, stateForWeightsAndBeta)
            assetPrev = asset
        return portValueCur

    @tf.function
    def objectiveFunction(self, nbSimul):
        toOptim = 0.
        for beta in self.betaList:
            portValueCur = self.simPorfolio(nbSimul, beta)
            Mean = tf.reduce_mean(portValueCur)
            mask = tf.less(portValueCur, tfp.stats.percentile(portValueCur, (1 - self.percentile) * 100))
            CVaR = -tf.reduce_mean(tf.boolean_mask(portValueCur, mask))
            toOptim = toOptim - Mean + beta * CVaR
        return toOptim

    @tf.function
    def trainStep(self, nbSimul):
        with tf.GradientTape() as tape:
            objFunc = self.objectiveFunction(nbSimul)
        gradients = tape.gradient(objFunc, self.kerasModel.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.kerasModel.trainable_variables))
        return objFunc

    def train(self, batchSize, numBatchDraws, numEpochs, weightsFolder):
        for epoch in range(numEpochs):
            start_time = time.time()
            for batchDraw in range(numBatchDraws):
                print(f'Draw Batch no.: {batchDraw}/{numBatchDraws} @ epoch {epoch}/{numEpochs}')
                objFuncAtTrainStep = self.trainStep(batchSize)
                self.loss.append(objFuncAtTrainStep)
            end_time = time.time()
            rtime = end_time - start_time
            print(f'Epoch {epoch} from overall {numEpochs} done in Time: {rtime}')
            self.linearDecrease(epoch, numEpochs)
        if isinstance(self.assetModel, BlackScholesModelWithBankAccount):
            pathForSavedWeights = f'{self.name}_batch={batchSize}_withRiskFreeRate={self.assetModel.r}.h5'
        else:
            pathForSavedWeights = f'{self.name}_batch={batchSize}_dim={self.kerasModel.dim}.h5'
        self.kerasModel.save_weights(os.path.join(weightsFolder, pathForSavedWeights))

    def simulateAccurateWholeFrontier(self, batchSizeEval, simulationsPerBeta):
        print('Simulation of Efficient Frontier for different beta')
        toOptimList = []
        MeanList = []
        CVaRList = []
        for i, beta in enumerate(self.betaList):
            print(f'beta {i}/{len(self.betaList)}')
            portVal = []
            for itime in range(simulationsPerBeta):
                portVal.append(self.simPorfolio(batchSizeEval, beta).numpy())  # get the number of simulations
            portVal = np.concatenate(portVal)
            Mean = np.mean(portVal)
            mask = np.less(portVal, tfp.stats.percentile(portVal, (1 - self.percentile) * 100))
            CVar = -np.mean(tf.boolean_mask(portVal, mask))
            # Objective Function
            toOptim = -Mean + beta * CVar
            # This is a list for different betas
            toOptimList.append(toOptim)
            MeanList.append(Mean)
            CVaRList.append(CVar)
        return np.array(MeanList), np.array(CVaRList), np.array(toOptimList)
