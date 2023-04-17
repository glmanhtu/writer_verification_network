import argparse

import matplotlib
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
import pandas as pd

# matplotlib.use('TkAgg')

parser = argparse.ArgumentParser()

parser.add_argument('--similarity_file', type=str, help='Path to the similarity file', required=True)

args = parser.parse_args()

# Load similarity matrix from CSV file
similarity_matrix = pd.read_csv(args.similarity_file, index_col=0)

# Perform hierarchical clustering
linkage_matrix = linkage(similarity_matrix, method='ward')

plt.figure(figsize=(6, 12))

# Plot dendrogram
dendrogram(linkage_matrix, orientation='right', labels=similarity_matrix.index)

plt.savefig("merged-dendrogram.pdf")
