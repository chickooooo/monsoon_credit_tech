import pandas as pd


def load_csv(file_path: str) ->  pd.DataFrame:
    """Loads CSV file and return it as a dataframe

    Args:
        file_path (str): path to the file

    Returns:
        pd.DataFrame: file converted to dataframe
    """

    data = pd.read_csv(file_path)
    # supressing pandas default behaviour
    data.rename(columns={'Unique_ID.1': 'Unique_ID'}, inplace=True)
    return data
