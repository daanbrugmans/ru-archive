import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class DataQualityAssessor:
    def __init__(self, dataset: pd.DataFrame) -> None:
        self.dataset = dataset
        
    def get_null_rates(self) -> pd.DataFrame:
        df_is_null = self.dataset.copy().isna()
        null_rates = {}
        
        for column_name in df_is_null.columns:
            row_count = df_is_null[column_name].size
            null_row_count = df_is_null[df_is_null[column_name] == True][column_name].size
            
            null_rates[column_name] = round(null_row_count / row_count, 2)
            
        null_rates_df = pd.DataFrame([null_rates]).T
        null_rates_df.columns = ["NULL Rate"]
            
        return null_rates_df
    
    def get_duplicate_rate(self, primary_key: list[str]) -> float:
        row_count = len(self.dataset)
        row_count_without_duplicates = len(self.dataset.drop_duplicates(subset=primary_key))
        
        return round(1 -  row_count_without_duplicates / row_count, 2)
    
    def draw_boxplots(self) -> None:
        sns.boxplot(data=pd.melt(self.dataset.select_dtypes(include="number")), x="value", y="variable")
        # plt.xscale("log")
        
    def draw_categorical_feature(self, feature_name: str) -> None:
        sns.countplot(data=self.dataset, y=feature_name, order=self.dataset[feature_name].value_counts().index)
        
    def draw_continuous_feature(self, feature_name: str) -> None:
        sns.histplot(data=self.dataset, x=feature_name)