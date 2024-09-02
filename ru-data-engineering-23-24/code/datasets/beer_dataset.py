from abc import ABC

import pandas as pd

class BeerDataset(ABC):
    def __init__(self) -> None:
        super().__init__()
    
        self.df: pd.DataFrame
        self.has_ratings: bool