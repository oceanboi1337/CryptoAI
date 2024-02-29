import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def main():
    df = pd.read_csv('datasets/formated.csv')
    numeric_df = df.select_dtypes(include=[np.number])
    correlation_matrix = numeric_df.corr()

    #close_correlations = correlation_matrix['close']

    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm')
    plt.show()

if __name__ == '__main__':
    main()