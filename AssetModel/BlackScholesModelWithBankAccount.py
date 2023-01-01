# Black Scholes model with risk-free asset. The implementation makes use of the Cholesky decomposition
# from which correlated Brownian Motions can be simulated.

import numpy as np
import tensorflow as tf


class BlackScholesModelWithBankAccount:
    def __init__(self, S0, mu, sigma, correl, bankAccount, r):
        self.S0 = S0
        self.mu = mu
        self.sigma = sigma
        self.corFac = np.linalg.cholesky(correl)
        self.correl = correl
        self.bankAccount = bankAccount
        self.r = r
        self.d = np.size(S0)
        self.name = 'BS_withRiskFree' if self.bankAccount else 'BS_OnlyRisky'

    def initValues(self):
        return self.S0

    def size(self):
        return np.shape(self.S0)[0]

    def average(self, t):
        return self.S0 * np.exp(self.mu * t)

    def std(self, t):
        return self.S0 * np.exp(self.mu * t) * np.sqrt(np.exp(self.sigma * self.sigma * t) - 1)

    # One time step with simple Euler-Scheme, using Cholesky decomposition for generating correlated BM.
    def oneStep(self, S, dt):
        muTF = tf.constant(self.mu, dtype=tf.float32)
        sigmaTF = tf.constant(self.sigma, dtype=tf.float32)
        corFacTF = tf.constant(self.corFac, dtype=tf.float32)
        # Extract the assets to differ between bank account and risky assets

        if self.bankAccount:
            # Bank Account; get the first colum
            bankAccount = S[:, 0]
            # Accrue bank account,
            multiplicatorRiskfree = tf.exp(self.r * dt)
            BankAccountTensorOut = tf.cast(bankAccount, tf.float32) * multiplicatorRiskfree
            BankAccountShaped = tf.reshape(BankAccountTensorOut, [tf.shape(BankAccountTensorOut)[0], 1])
            SRiskyAssets = S[:, 1:tf.cast(S, tf.float32).get_shape()[1]]  # [1] because here we extract the columns

            # Risky assets
            # Cholesky Decomposition for correlated Brownian Motion
            randVar = tf.einsum("ij,lj->li", corFacTF, tf.random.normal(shape=tf.shape(SRiskyAssets)))
            # Next step by simple euler scheme
            SoutRiskyAsset = SRiskyAssets * tf.exp((muTF - self.sigma * self.sigma / 2) * dt +
                                                   tf.einsum("ij,j->ij", randVar, sigmaTF * np.sqrt(dt)))
            # Cast it to tf.float32
            SoutRiskyAssetTensor = tf.cast(SoutRiskyAsset, tf.float32)

            # Put Bank account together with Risky assets
            Sout = tf.concat([BankAccountShaped, SoutRiskyAssetTensor], 1)
        else:
            randVar = tf.einsum("ij,lj->li", corFacTF, tf.random.normal(shape=tf.shape(S)))
            Sout = S * tf.exp(
                (self.mu - self.sigma * self.sigma / 2) * dt + tf.einsum("ij,j->ij", randVar, sigmaTF * np.sqrt(dt)))
        return Sout
