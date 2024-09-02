import datasets.untappd
import data_wrangler as dw

import warnings

import pandas as pd


warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

end_user_dataset = datasets.untappd.UntappdDataset()
data_wrangler = dw.DataWrangler(end_user_dataset)

data_wrangler.handle_duplicates()
data_wrangler.parse_category_and_type_features()
data_wrangler.generate_flavor_features()

data_wrangler.save_as_parquet("end_user_dataset")