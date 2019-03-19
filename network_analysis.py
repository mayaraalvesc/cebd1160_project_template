#!/usr/bin/env python

from argparse import ArgumentParser
import os.path as op
import os

import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize as nmlz
from sklearn.linear_model import Ridge

import warnings
warnings.filterwarnings("ignore")
# Bad practice, don't do the above :)


def prepare_data(graph_dir, nodes):
    # First, get file paths for all of the graphs
    # (P.s. a graph == a network, not a plot/figure)
    graph_files = [op.join(graph_dir, gfile)
                   for gfile in os.listdir(graph_dir)]

    # Then, initialize an empty matrix to store them all
    graph_data = np.empty((len(graph_files), nodes, nodes))

    # For each file...
    for idx, single_graph in enumerate(graph_files):
        # Grab the data
        tmp_graph = np.loadtxt(single_graph)

        # Create an empty matrix to populate
        adjacency = np.reshape(np.repeat(np.nan, nodes*nodes),
                               (nodes, nodes))

        # Populate the matrix with the reorganized data in the upper triangle
        for row in tmp_graph:
            i, j = int(row[0]-1), int(row[1]-1)
            adjacency[i, j] = float(row[2])

        # Add the newly organized data to the main data structure
        graph_data[idx, :, :] = adjacency
    return graph_data


def plot_graph(graph, anno_locs=None, anno_text=None, fname=False, title=None):
    plt.clf()
    # Plain and simple, plot the graph on a log scale
    # (add 1 so that zeros don't make things break)
    plt.imshow(np.log10(graph+1))
    plt.colorbar()

    # Adds a title
    if title:
        plt.title(title)

    # If we've provided annotations, add them to the plot
    if anno_locs and anno_text:
        for val, (i, j) in zip(anno_text, anno_locs):
            plt.text(j, i, val)

    # If specified, expecting the "fname" parameter to be a filename
    if fname:
        plt.savefig(fname)
    # If not specified, just show us the figure instead
    else:
        plt.show()


def model_graphs(graph_table, normalize=True):
    np.random.seed(123457)
    if normalize:
        graph_table = nmlz(graph_table, axis=1)
    # Extract the columns of interest
    X = graph_table[:,0:12]
    y = graph_table[:,-1]

    # Generate training and testing set
    # N.B.: I did not do cross validation, just an ordinary split, but should:)
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y)

    # Establish model and params
    # N.B.: Offline I played with parameters and picked these, but didn't bake
    # that in to this script.
    model = Ridge(0.9)
    model.fit(Xtrain, ytrain)

    return model.score(Xtest, ytest), model.predict(Xtest), ytest


def plot_performance(predicted, expected, fname=None, title=None):
    plt.clf()
    # Add reference line for true weights relative to off-guesses
    plt.axhline(np.mean(expected), color="blue")
    plt.axhline(np.mean(predicted), color="orange")

    # Show a jittered scatter plot for all performance numbers and their errors
    xvals = np.linspace(0, 1, len(predicted))
    plt.scatter(xvals, expected, color="blue")
    plt.scatter(xvals, predicted, color="orange")
    plt.scatter(xvals, np.abs(predicted-expected), color="red")

    plt.ylabel("Connection Strength")
    plt.xlabel("Testing Sample")
    plt.xticks([])
    plt.legend(["Average True", "Average Predicted", "True", "Predicted",
                "Absolute Prediction Error"])
    plt.axhline(0, color='black', linestyle='--', linewidth=0.5)

    # Adds a title
    if title:
        plt.title(title)

    # If specified, expecting the "fname" parameter to be a filename
    if fname:
        plt.savefig(fname)
    # If not specified, just show us the figure instead
    else:
        plt.show()


def main():
    parser = ArgumentParser()
    parser.add_argument("graph_directory",
                        help="Directory containing the graph files.")
    parser.add_argument("output_directory", default=os.getcwd(),
                        help="Directory where outputs will be saved. "
                             "Defaults to current working directory")
    parser.add_argument("--nodes", "-n", type=int,
                        default=6,
                        help="The number of regions in the graphs. "
                             "Defaults to 6.")
    args = parser.parse_args()

    ##########################
    #                        #
    # Step 1: Data Grooming  #
    #                        #
    ##########################

    # Load graphs and compute an average, ignoring NaNs
    graphs = prepare_data(args.graph_directory, args.nodes)
    average_graph = np.nanmean(graphs, axis=0)

    # Find all the locations that have edges in the graph
    locs = list(zip(*np.where(~np.isnan(average_graph))))

    # Plot the graphs and location list
    fname = op.join(args.output_directory, "average_graph.png")
    plot_graph(average_graph, anno_locs=locs, anno_text=range(len(locs)),
               fname=fname, title="Average graph with feature locations")

    # Create an empty table of the right shape
    graph_table = np.zeros((len(graphs), len(locs)))

    # Put data in the positions defined by "locs" into the table
    for row, graph in enumerate(graphs):
        graph_table[row, :] = [graph[l] for l in locs]

    # Rest all NaNs to 0s now that the data has been pruned
    graph_table[np.where(np.isnan(graph_table))] = 0

    ##########################
    #                        #
    # Step 2: Data Modeling  #
    #                        #
    ##########################
    # Ok, so I happen to know that regions 0, 1, 2 are the LH and correspond to
    # 3, 4, and 5 in the RH. So, from the connections between 0 -> 1, 0 -> 2,
    # can I predict the connections between 3 -> 4, 3 -> 5?
    #
    # Possibly easier starting point: can I predict the connections between
    # 4 -> 5 from all the other connections?
    #
    # From my earlier figure, I know this means I am trying to predict col 12
    # using all the others.

    # Train and evaluate the model
    score, predicted, expected = model_graphs(graph_table, normalize=True)
    print("Performance of classifier (R^2 error): {0}".format(score))

    # Plot the residual edge weight
    fname = op.join(args.output_directory, "performance.png")
    title = "Brain Connectivity Prediction (R^2={0})".format(score)
    plot_performance(predicted, expected, fname=fname, title=title)


if __name__ == "__main__":
    main()
