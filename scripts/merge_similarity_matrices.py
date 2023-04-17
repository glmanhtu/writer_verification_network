import argparse

import pandas as pd

parser = argparse.ArgumentParser()

parser.add_argument('--similarity_files', type=str, help='Specify all similarity matrices to merge', required=True,
                    nargs='+')

args = parser.parse_args()

dataframes = {}
categories = set([])
for file in args.similarity_files:
    df = pd.read_csv(file, index_col=0)
    dataframes[file] = df
    categories = set(df.columns) if len(categories) == 0 else set(categories & set(df.columns))

categories = sorted(list(categories))
for df in dataframes:
    # Use only the common categories between similarity matrix files
    dataframes[df] = dataframes[df].loc[categories, categories]

# Merge the dataframes together by taking the average of the values for each category
merged_df = pd.concat(dataframes.values()).groupby(level=0).mean()

# Write the merged dataframe to a new CSV file
merged_df.to_csv('merged_similarity.csv')
