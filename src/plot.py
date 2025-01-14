#
# Plot functions for post processing for novel constrained sequential Latin Hypercube (with multidimensional uniformity) method
# Jun 2024
# author: Christina Schenk

#Python Packages:
import matplotlib.pyplot as plt
from itertools import combinations
from sklearn.decomposition import PCA
import seaborn as sns
import numpy as np
import pandas as pd
import math
plt.rcParams.update({'font.size': 14})
#---------------------------------------------------------------------------------------------------#
#### Plotting functions for post processing new designs and analyzing uniformity etc.:


def plot_dimred_2dims_both_methods(data_pca,
                                   lhs_samples_pca,
                                   lhsmdu_samples_pca,
                                   filename_eps=''):
    
    """
    generates scatter plot of data conditioned LHS and conditioned LHSMDU samples

    Parameters
    ----------
    data: np array of original data
    lhs_samples: np array of LHS samples
    lhsmdu_samples: np array of LHSMDU samples
    filename_eps: string of path with eps filename

    Returns
    -------
    Scatterplot

    """
    plt.scatter(data_pca[:,0], data_pca[:,1])
    plt.scatter(lhs_samples_pca[:,0], lhs_samples_pca[:,1])
    plt.scatter(lhsmdu_samples_pca[:,0], lhsmdu_samples_pca[:,1])
    plt.legend(['Data', 'LHS', 'LHSMDU'])
    if len(filename_eps)>0:
        plt.savefig(filename_eps, format='eps')
        plt.show()
    else:
        plt.show()


def distplot_samples(samples,
                     filename_eps=''):
    """
    generates distribution kde plot of samples

    Parameters
    ----------
    samples: np array of samples nsamp x ncomponents
    filename_eps: string of path with eps filename

    Returns
    -------
    Distplot with distributions for different components in different colors

    """
    ax = sns.displot(samples,
    kind="kde")
    ax.set(xlabel='Distribution', ylabel='Density')
    if len(filename_eps)>0:
        plt.savefig(filename_eps, format='eps')
        plt.show()
    else:
        plt.show()

        
def box_kdeplot_samples(samples,
                        filename_eps='',
                        fixed_ranges=None):
    """
    generates distribution box kde subplots of samples

    Parameters
    ----------
    samples: np array of samples nsamp x ncomponents
    filename_eps: string of path with eps filename

    Returns
    -------
    Subplots showing box kde distributions

    """
    if samples.shape[1]%2==0:
        cols = samples.shape[1]//2
    elif samples.shape[1]%3==0:
        cols = samples.shape[1]//3
    else:
        cols = samples.shape[1]//3
    rows = math.ceil(samples.shape[1]/cols)
    fig, axes = plt.subplots(cols, rows)
    axes = axes.ravel()  # flattening the array makes indexing easier
    for col, ax in zip(range(samples.shape[1]), axes):
        sns.histplot(data = samples[:,col],kde=True, stat='density', ax=ax)
        ax.set(xlabel="component "+str(col), ylabel='Density')
        
        # Set fixed ranges if provided
        if fixed_ranges is not None and col < len(fixed_ranges):
            ax.set_xlim(fixed_ranges[col])
            
    fig.tight_layout()
    if len(filename_eps)>0:
        plt.savefig(filename_eps, format='eps')
        plt.show()
    else:
        plt.show()

        
def create_pairwise_scatterplots(data,
                                 lhs,
                                 lhsmdu,
                                 dim_labels=None,
                                 colors=None,
                                 labels=None,
                                 figsize=(15, 10),
                                 filename_eps="",
                                 plots_per_fig=9):
    """
    Create pairwise scatterplots for given datasets, splitting into multiple figures if necessary.

    Parameters:
    -----------
    data: np array of original data
    lhs_samples: np array of LHS samples
    lhsmdu_samples: np array of LHSMDU samples
    dim_labels: list of str, labels for each dimension. Defaults to 'component X'.
    colors: list of str, colors for each dataset. Defaults to ['blue', 'orange', 'green'].
    labels: list of str, labels for each dataset. Defaults to ['Data', 'LHS', 'LHSMDU'].
    figsize: tuple, size of each figure. Defaults to (15, 10).
    filename_prefix: prefix for filenames when saving figures.
    plots_per_fig: int, number of plots per figure.
    """
    if dim_labels is None:
        dim_labels = [f'component {i+1}' for i in range(data.shape[1])]
    if colors is None:
        colors = ['blue', 'orange', 'green']
    if labels is None:
        labels = ['Data', 'LHS', 'LHSMDU']
    # Create pairwise combinations of dimensions
    pairs = list(combinations(range(data.shape[1]), 2))
    total_plots = len(pairs)
    n_figs = (total_plots + plots_per_fig - 1) // plots_per_fig  # Total number of figures

    for fig_idx in range(n_figs):
        # Create a new figure for this batch
        fig, axes = plt.subplots((plots_per_fig + 2) // 3, 3, figsize=figsize)
        axes = axes.flatten()

        # Plot the subset of pairs for this figure
        for plot_idx, (dim1, dim2) in enumerate(pairs[fig_idx * plots_per_fig : (fig_idx + 1) * plots_per_fig]):
            ax = axes[plot_idx]
            ax.scatter(data[:, dim1], data[:, dim2], color=colors[0], alpha=0.6, label=labels[0])
            ax.scatter(lhs[:, dim1], lhs[:, dim2], color=colors[1], alpha=0.6, label=labels[1])
            ax.scatter(lhsmdu[:, dim1], lhsmdu[:, dim2], color=colors[2], alpha=0.6, label=labels[2])

            ax.set_xlabel(dim_labels[dim1])
            ax.set_ylabel(dim_labels[dim2])
            ax.set_title(f"Pair: {dim_labels[dim1]} vs {dim_labels[dim2]}")
            ax.legend()

        # Hide unused subplots
        for plot_idx in range(len(pairs[fig_idx * plots_per_fig : (fig_idx + 1) * plots_per_fig]), len(axes)):
            axes[plot_idx].axis('off')

        # Adjust layout
        plt.tight_layout()
        if len(filename_eps)>0:
            plt.savefig(filename_eps, format='eps')
            plt.show()
        else:
            plt.show()
            
            
def create_pairwise_distribution_plots_seaborn(data,
                                               lhs,
                                               lhsmdu,
                                               markers=None,
                                               dim_labels=None,
                                               labels=None,
                                               filename_eps=""):
    """
    Create pairwise distribution plots using Seaborn.

    Parameters:
    -----------
    data: np.array, original dataset.
    lhs: np.array, LHS samples.
    lhsmdu: np.array, LHSMDU samples.
    dim_labels: list of str, labels for each dimension. Defaults to 'dim i'.
    labels: list of str, dataset labels. Defaults to ['Data', 'LHS', 'LHSMDU'].
    filename_eps: str, file path for saving the plot as EPS.
    """
    if dim_labels is None:
        dim_labels = [f'dim {i+1}' for i in range(data.shape[1])]
    if labels is None:
        labels = ['Data', 'LHS', 'LHSMDU']

    # Combine data into a DataFrame
    data_df = pd.DataFrame(data, columns=dim_labels)
    lhs_df = pd.DataFrame(lhs, columns=dim_labels)
    lhsmdu_df = pd.DataFrame(lhsmdu, columns=dim_labels)

    # Add labels for datasets
    data_df['Dataset'] = labels[0]
    lhs_df['Dataset'] = labels[1]
    lhsmdu_df['Dataset'] = labels[2]

    # Combine all datasets
    combined_df = pd.concat([data_df, lhs_df, lhsmdu_df])
    
    # Create pairplot with a custom palette
    palette = {'Data': 'blue', 'LHS': 'orange', 'LHSMDU': 'green'}

    # Create pairplot
    if markers==None:
        pairplot = sns.pairplot(combined_df, hue='Dataset', diag_kind='kde', palette=palette)
    else: 
        pairplot = sns.pairplot(combined_df, hue='Dataset', markers=markers, diag_kind='kde', palette=palette)
    if filename_eps:
        pairplot.savefig(filename_eps)
    plt.show()
