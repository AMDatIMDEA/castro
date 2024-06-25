#
# Plot functions for post processing for novel constrained sequential Latin Hypercube (with multidimensional uniformity) method
# Jun 2024
# author: Christina Schenk

#Python Packages:
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns
import numpy as np
import pandas as pd
import math

#---------------------------------------------------------------------------------------------------#
#### Plotting functions for post processing new designs and analyzing uniformity etc.:


def plot_dimred_2dims_both_methods(data_pca, lhs_samples_pca, lhsmdu_samples_pca):
    """
    generates scatter plot of data conditioned LHS and conditioned LHSMDU samples

    Parameters
    ----------
    data: integer of dimension of input space
    lhs_samples: list of upper and lower bounds
    lhsmdu_samples: string of method: LHS or LHSMDU

    Returns
    -------
    Scatterplot

    """
    plt.scatter(data_pca[:,0], data_pca[:,1])
    plt.scatter(lhs_samples_pca[:,0], lhs_samples_pca[:,1])
    plt.scatter(lhsmdu_samples_pca[:,0], lhsmdu_samples_pca[:,1])
    plt.legend(['Data', 'LHS', 'LHSMDU'])
    plt.show()


def distplot_samples(samples):
    """
    generates distribution kde plot of samples

    Parameters
    ----------
    samples: np array of samples nsamp x ncomponents

    Returns
    -------
    Distplot with distributions for different components in different colors

    """
    ax = sns.displot(samples,
    kind="kde")
    ax.set(xlabel='Distribution', ylabel='Density')
    plt.show()

def box_kdeplot_samples(samples):
    """
    generates distribution box kde subplots of samples

    Parameters
    ----------
    samples: np array of samples nsamp x ncomponents

    Returns
    -------
    Subplots showing box kde distributions

    """
    cols = samples.shape[1]//2
    rows = math.ceil(samples.shape[1]/cols)
    fig, axes = plt.subplots(cols, rows)
    axes = axes.ravel()  # flattening the array makes indexing easier
    for col, ax in zip(range(samples.shape[1]), axes):
        sns.histplot(data = samples[:,col],kde=True, stat='density', ax=ax)
        ax.set(xlabel="component "+str(col), ylabel='Density')
    fig.tight_layout()
    plt.show()
