from datasets.beer_dataset import BeerDataset

from pathlib import Path

import pandas as pd

class DataWrangler:
    def __init__(self, dataset: BeerDataset) -> None:
        self.dataset = dataset
        self.dataset.df = self.dataset.df[["beer_name", "brewery_name", "beer_type", "beer_abv", "beer_ibu", "comment", "flavor_profiles", "rating_score"]]
        self.dataset.df = self.dataset.df.rename(columns={
            "beer_name": "beer_name",
            "brewery_name": "brewery_name",
            "beer_type": "beer_type",
            "beer_abv": "abv",
            "beer_ibu": "ibu",
            "comment": "description",
            "flavor_profiles": "flavor_profile",
            "rating_score": "rating"
        })
    
    def handle_duplicates(self):       
        grouped_df = self.dataset.df.groupby(by=["beer_name", "brewery_name", "beer_type", "abv", "ibu"], as_index=False)
        grouped_df = grouped_df.agg({
            "description": lambda x: pd.NA if pd.isna(x.all()) else ("; ".join(x.dropna().to_list()).replace("\n", ". ") if ("".join(x.dropna().to_list()) != "") else pd.NA),
            "rating": "mean",
            "flavor_profile": lambda x: pd.NA if pd.isna(x.all()) else ",".join(x.dropna().to_list())
        })
        
        self.dataset.df = grouped_df
    
    def generate_flavor_features(self):
        found_flavors = set()
        
        for _, row in self.dataset.df.iterrows():
            if pd.isna(row.flavor_profile):
                continue
            
            beer_flavor_profile = set(row.flavor_profile.split(","))
            found_flavors = found_flavors.union(beer_flavor_profile)
            
        found_flavors = sorted(found_flavors)
            
        for flavor in found_flavors:
            self.dataset.df[flavor] = False
            
        for index, (_, row) in enumerate(self.dataset.df.iterrows()):
            if pd.isna(row.flavor_profile):
                continue
            
            beer_flavor_profile = row.flavor_profile.split(",")
            
            for flavor in beer_flavor_profile:
                self.dataset.df.loc[index, flavor] = True
            
        self.dataset.df = self.dataset.df.drop(["flavor_profile"], axis=1)
    
    def parse_category_and_type_features(self):
        self.dataset.df["beer_category"] = ""
        self.dataset.df.insert(2, "beer_category", self.dataset.df.pop("beer_category"))
        
        for index, (_, row) in enumerate(self.dataset.df.iterrows()):
            if " - " in row.beer_type:
                category_name = row.beer_type.split(" - ")[0].strip()
                type_name = row.beer_type.split(" - ")[1].strip()
                
                self.dataset.df.loc[index, "beer_category"] = category_name
                self.dataset.df.loc[index, "beer_type"] = type_name + " " + category_name
            else:
                self.dataset.df.loc[index, "beer_category"] = row.beer_type
    
    def save_as_parquet(self, filename: str):
        if str(Path.cwd()).endswith("code"):
            cwd = Path.cwd().parents[0]
        else:
            cwd = Path.cwd()
        
        path_to_parquet_file = Path.joinpath(cwd, "data", "wrangled", f"{filename}.parquet")
        
        self.dataset.df.to_parquet(path_to_parquet_file)
        
        print(f"Saved dataset as parquet file to {path_to_parquet_file}")