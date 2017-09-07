import pandas as pd


def open_datasets(origin_filename, dest_filename):
    """
    Merely takes in filenames, and returns
    dataframes from those filenames

    Data is expected as csv
    """
    origin = pd.read_csv(origin_filename)
    dest = pd.read_csv(dest_filename)
    return (origin, dest)

if __name__ == "__main__":
    open_datasets('crimes2015.csv', 'crimes2015.csv')
