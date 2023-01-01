import os
import sys

import numpy as np
from scipy.stats import random_correlation


def create_correlation_risky_assets(nDim, bankAccount, csvFolder, seed):
    np.random.seed(seed)
    if nDim > 1:
        eigvectorCor = np.random.uniform(low=0, high=1, size=nDim)
        eigvectorCor = nDim * (eigvectorCor / eigvectorCor.sum())
        corr = random_correlation.rvs(eigvectorCor, random_state=seed)
    else:
        corr = np.ones([1, 1])
    np.savetxt(os.path.join(csvFolder, f'correlation_risky_assets_bA={bankAccount}_nDimRisky={nDim}_seed={seed}.csv'), corr, fmt='%f')
    return corr


def validate_correlation_risky_assets(nDim, corr):
    for i in range(nDim):
        if np.fabs(corr[i, i] - 1) > 1e-10:
            print("diag should be 1")
            sys.exit(0)
        for j in range(i):
            if np.fabs(corr[i, j] - corr[j, i]) > 1e-10:
                print("Non Sym")
                sys.exit(0)

    maxCorr = np.amax(np.abs(corr))
    if maxCorr > 1 + 1e-9:
        print(" Should be  1 ", maxCorr)
        sys.exit(0)
    else:
        # print("extra diag below 1")
        pass
