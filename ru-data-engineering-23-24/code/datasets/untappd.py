import datasets.beer_dataset

from pathlib import Path

import pandas as pd

class UntappdDataset(datasets.beer_dataset.BeerDataset):
    def __init__(self) -> None:
        if str(Path.cwd()).endswith("code"):
            cwd = Path.cwd().parents[0]
        else:
            cwd = Path.cwd()
        
        self.path_to_csv = Path.joinpath(cwd, "data", "raw", "untappd", "check-ins.csv")
        self.df = pd.read_csv(self.path_to_csv)
        self.has_ratings = True
        
