import os

from matplotlib import pyplot as plt

from AssetModel import asset_model_utils
from AssetModel.BlackScholesModelWithBankAccount import BlackScholesModelWithBankAccount


def get_asset_model(nbRiskyAssets, S0, mu, sigma, corr, bankAccount, r=0.1):
    # Validate that your correlation is correct
    asset_model_utils.validate_correlation_risky_assets(nbRiskyAssets, corr)
    return BlackScholesModelWithBankAccount(S0, mu, sigma, corr, bankAccount, r)


def evaluate_optimizer(optimizer, batchSizeEval, simulationsPerBeta=200):
    expOpt, cVaROpt, optimList = optimizer.simulateAccurateWholeFrontier(batchSizeEval,
                                                                         simulationsPerBeta)

    return expOpt, cVaROpt, optimList


def plot_frontier(expOpt, cVaROpt, pgf):
    if pgf:
        import matplotlib
        matplotlib.use("pgf")
        matplotlib.rcParams.update({
            "pgf.texsystem": "pdflatex",
            'font.family': 'serif',
            'font.size': 14,
            'text.usetex': True,
            'pgf.rcfonts': False,
        })
    fig, ax = plt.subplots()
    ax.plot(cVaROpt, expOpt, linestyle='dashed')
    ax.set_title('Efficient Frontier')
    ax.set_xlabel('Conditional Value at Risk')
    ax.set_ylabel("Expectation")
    plt.tight_layout()
    return ax


def plot_loss(optimizer, pgf):
    if pgf:
        import matplotlib
        matplotlib.use("pgf")
        matplotlib.rcParams.update({
            "pgf.texsystem": "pdflatex",
            'font.family': 'serif',
            'font.size': 14,
            'text.usetex': True,
            'pgf.rcfonts': False,
        })
    fig, ax = plt.subplots()
    ax.plot(optimizer.loss, linestyle='dashed')
    ax.set_title('Loss')
    plt.tight_layout()
    return ax


def save_frontier(expOpt, cVaROpt, plotFolder, plotName):
    plot_frontier(expOpt, cVaROpt, pgf=False)
    plt.savefig(os.path.join(plotFolder, f'{plotName}.png'), bbox_inches='tight')
    plot_frontier(expOpt, cVaROpt, pgf=True)
    plt.savefig(os.path.join(plotFolder, f'{plotName}.pgf'), bbox_inches='tight')


def save_loss(optimizer, plotName, plotFolder):
    plot_loss(optimizer, pgf=False)
    plt.savefig(os.path.join(plotFolder, f'{plotName}.png'), bbox_inches='tight')
    plot_loss(optimizer, pgf=True)
    plt.savefig(os.path.join(plotFolder, f'{plotName}.pgf'), bbox_inches='tight')
