from pathlib import Path

import pandas as pd

class OpenBeerDBDataset:
    def __init__(self) -> None:
        query = """
        SELECT beers.name AS beer_name, breweries.name AS brewery_name, categories.cat_name AS category, styles.style_name AS style, beers.abv, beers.ibu, beers.descript
        FROM beers
        INNER JOIN breweries
            ON beers.brewery_id = breweries.id
        INNER JOIN categories
            ON beers.cat_id = categories.id
        INNER JOIN styles
            ON beers.style_id = styles.id
        """
        
        if str(Path.cwd()).endswith("code"):
            cwd = Path.cwd().parents[0]
        else:
            cwd = Path.cwd()
        
        # self.path_to_csv = Path.joinpath(cwd, "data", "untappd", "check-ins.csv")
        self.df = pd.read_csv(self.path_to_csv)