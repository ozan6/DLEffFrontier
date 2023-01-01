# Run test classes using trained weights (on LRZ Linux Cluster). We need the plots and results.
import os
from pathlib import Path

import numpy as np
import tensorflow as tf

from AssetModel import asset_model_utils
from MeanCVaROptimizer.MeanCVaROptimizer import MeanCVaROptimizer
from NeuralNetworks.FeedforwardNeuralNetwork import FeedForwardWithPositiveAndNormedWeightsCreator
from utils import evaluate_optimizer, get_asset_model, save_frontier, save_loss

# We need the following objects:  -Neural Network with all parameters (No. of layers, activation functions, optimizer etc.)
#                                  - AssetModel with all Parameters (Initialvalue, drift, correlation, volatility)
#                                  - Simulation with all parameters (No. of Sim)
#                                  - Optimizer, combining AssetModel, Simulation and Neural Network to yield Objective Function
#                                                           and trainable optimizer as well

seed = 10
np.random.seed(seed)

weightsFolder = "savedWeights"
plotFolder = "plots"
csvFolder = "csv"

Path(weightsFolder).mkdir(parents=True, exist_ok=True)
Path(plotFolder).mkdir(parents=True, exist_ok=True)
Path(csvFolder).mkdir(parents=True, exist_ok=True)


def run_main(train, pathForWeights=None):
    ######################

    # Set relevant parameters

    ######################

    # Set the number of assets first and initialize
    nbAssets = 2
    S0 = np.ones(nbAssets)

    # Percentile for Conditional Value At Risk
    percentile = 0.95

    # Initial portfolio
    portValueInit = 1.

    # Set time horizon and rebalancing times (times for discretization as well)
    T = 5  # 5 years
    dt = 1. / 12  # each month

    ######################

    # Create asset model

    ######################

    # Set Bank Account
    bankAccount = True

    if bankAccount:
        nbRiskyAssets = nbAssets - 1
        r = 0.1
    else:
        nbRiskyAssets = nbAssets
        r = None

    # Risky Asset Parameters (Initial value, Mu, Sigma)
    corr = asset_model_utils.create_correlation_risky_assets(nbRiskyAssets, bankAccount, csvFolder, seed)
    mu = np.arange(nbRiskyAssets) / (nbRiskyAssets + 5 * 20) + 0.1
    sigma = 0.1 * np.arange(nbRiskyAssets) / nbRiskyAssets + 0.05

    # Get asset model
    assetModel = get_asset_model(nbRiskyAssets, S0, mu, sigma, corr, bankAccount, r)

    ######################

    # Set optimizer variables

    ######################

    initialLR = 1E-2 / nbRiskyAssets
    finalLR = initialLR / 10

    # Network parameters
    activationFunction = tf.nn.tanh
    nbNeurons = 10 + nbRiskyAssets
    nbLayer = 3
    layerSize = np.ones([nbLayer]) * nbNeurons
    batchSize = 300
    batchSizeEval = 6000

    numBatchDraws = 50
    numEpochs = 50

    # Number of points on the Efficient Frontier
    nBeta = 40
    betaList = np.square(np.arange(0., 2, 2 / nBeta))

    ######################

    # Create keras model

    ######################
    kerasCreator = FeedForwardWithPositiveAndNormedWeightsCreator(nbRiskyAssets, layerSize, activationFunction)
    kerasModel = kerasCreator.create()

    if not train:
        inputs = tf.ones([batchSize, 3])  # Inputs are (t,x,\beta)
        kerasModel(inputs=inputs)
        if not pathForWeights:
            print('In order to evaluate a pretrained model, please specify a path to load the weights.')
        elif not os.path.isfile(pathForWeights):
            print('Could not find the file you specified as pathForWeights.')
        else:
            kerasModel.load_weights(pathForWeights)

    optimizer = MeanCVaROptimizer(percentile, assetModel, kerasModel, betaList, portValueInit,
                                  T, dt, initialLR, finalLR)

    if train:
        optimizer.train(batchSize, numBatchDraws, numEpochs, weightsFolder)

    expOpt, cVaROpt, optimList = evaluate_optimizer(optimizer, batchSizeEval)

    plot_name = f'{assetModel.name}_dim={nbAssets}_b={batchSize}_bE={batchSizeEval}_nbBatchDraw={numBatchDraws}_nbEpochs={numEpochs}_nBeta={nBeta}'

    save_frontier(expOpt, cVaROpt, plotName=plot_name, plotFolder=plotFolder)
    save_loss(optimizer, plotName=f'{plot_name}_loss', plotFolder=plotFolder)


run_main(train=True)
