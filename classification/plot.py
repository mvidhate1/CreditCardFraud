import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.pylab as pl

def plot_graph(df, feature, type, title, additional=None):
    '''
    Function for plotting the various kinds of graphs
    :param df: dataframe to be used
    :param feature: features from the dataframe to be used
    :param type: type of graph
    :return: null
    '''
    if type[0] == 'heatmap':
        if type[1] == 'corr':
            mask = np.triu(np.ones_like(df.corr(), dtype=np.bool_))
            plot = df.corr()
            print(df.groupby(feature).count().index, '\n', plot)
        sns.heatmap(plot,
                    annot=True,
                    mask=mask)

    elif type[0] == 'pairplot':
        sns.pairplot(df[feature],
                     hue=type[1])

    elif type[0] == 'line':
        ax = sns.lineplot(type[1], type[2],
                          marker="o")
        ax.lines[0].set_linestyle('--')
        plt.xlabel('k - value')
        plt.ylabel('Accuracy')

    elif type[0] == 'scatter':
        plt.scatter(df[feature[0]], df[feature[1]],
                    c=df[feature[2]].to_numpy())
        plt.xlabel(feature[0])
        plt.ylabel(feature[1])

    if type[1] == 'addlines':
        colors = pl.cm.jet(np.linspace(0, 1, len(additional)))
        for i in range(len(additional)):
            plt.plot([additional[i][0][0], additional[i][1][0]], [additional[i][0][1], additional[i][1][1]],
                 'k-',
                 lw=1.5,
                 c=colors[i])

    plt.suptitle(title)
    plt.show()


