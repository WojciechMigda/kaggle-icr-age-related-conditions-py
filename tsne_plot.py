#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def tsne_plot(X, target, *, random_state, perplexity=30):
    """
    Examples
    --------
    >>> from pyicr import tsne_plot
    >>> tsne_plot(df_ml_X_ii2_2, subclasses, perplexity=30, random_state=SEED)
    """
    tsne = TSNE(n_components=2, random_state=random_state, perplexity=perplexity, method='exact')
    X_2d = tsne.fit_transform(X)

    color_map = dict(enumerate(
        #['deepskyblue', 'skyblue', 'orangered', 'coral', 'seagreen', 'mediumseagreen', 'grey', 'lightgrey']
        #['deepskyblue', 'orangered', 'seagreen', 'crimson', 'olivedrab', 'purple']
        ['tab:red', 'tab:olive', 'tab:green', 'tab:blue', 'tab:brown', 'purple']
        ))

    plt.figure()
    for i, j in enumerate(np.unique(target)):
        plt.scatter(
            x=X_2d[target == j, 0],
            y=X_2d[target == j, 1],
            c=color_map[i],
            label=j,
            alpha=0.4
        )

    # Plot settings
    plt.xlabel('x')                              # X-axis label
    plt.ylabel('y')                              # Y-axis label
    plt.legend(loc='upper left')                 # Legend
    plt.title(f't-SNE plot\nperplexity={perplexity}')      # Title
    plt.show()
