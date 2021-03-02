import numpy as np
import pandas as pd



def main():
    returns = pd.read_csv('returns.txt', index_col=0)
    volumes = pd.read_csv('volumes.txt', index_col=0)
    prices = pd.read_csv('prices.txt', index_col=0)

    T, n = returns.shape
    print(T, n)

if __name__ == "__main__":
    # execute only if run as a script
    main()