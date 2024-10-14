import sys
sys.path += ["../src"]
import climact_shared.src.utils as cu
import sociodemographic_features
import pandas as pd
import numpy as np
from glob import glob
import spark_init
import pyspark.sql.functions as F
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from functools import reduce

sc = spark_init.spark_context()
sparkSQL = spark_init.SQLContext(sc)

attributes = ["age", "gender", "partisan", "affluence"]
subreddit_names_ = np.load("../data/list_subreddits.npy", allow_pickle=True)
scores_nature_ = pd.read_csv("../data/scores.big.csv")[["community"] + attributes]

subreddit_intersection = set(scores_nature_["community"])&set(subreddit_names_)
subreddit_names = np.array([u for u in subreddit_names_ if u in subreddit_intersection])
scores_nature = scores_nature_.query("community in @subreddit_intersection").set_index("community").loc[subreddit_names, :].reset_index()

all_dates = pd.date_range(start = pd.to_datetime("2015-03-01"), end = pd.to_datetime("2023-01-01"))
all_months = [u.strftime("%Y-%m") for u in all_dates[all_dates.day == 1]]

sociodemo = {attribute: sociodemographic_features.Nature_model(scores_nature[["community",attribute]], subreddit_names, attribute) 
             for attribute in attributes}

for attribute in attributes:
    sociodemo[attribute].fit()

def get_scores_month(month, attribute = None, save = True):
    count_subreddits_author_spark = (sparkSQL.read.parquet(cu.data_path + f"histories_strong_activated_and_control/RC_{month}.parquet")
                                     .filter(F.col("subreddit").isin(list(subreddit_names)))
                                     .groupby(["author", "subreddit"])
                                     .count()
                                     .toPandas()
                                     .set_index(["author", "subreddit"])
                                     .unstack()["count"]
                                     .reindex(columns = subreddit_names)
                                     [subreddit_names]
                                     .fillna(0)
                                     )
    
    if attribute is not None:
        scores_df = sociodemo[attribute].get_attribute_scores_df(count_subreddits_author_spark, attribute)
        
        if save: 
            scores_df.to_csv(cu.data_path + f"sociodemographic_features/{month}_{attribute}.csv.gz", compression = "gzip")
        else:
            return scores_df
    else:
        for attribute in attributes:
            scores_df = sociodemo[attribute].get_attribute_scores_df(count_subreddits_author_spark, attribute)
            if save:
                scores_df.to_csv(cu.data_path + f"sociodemographic_features/{month}_{attribute}.csv.gz", compression = "gzip")

if __name__ == "__main__":
    for month in all_months:
        print(month)
        get_scores_month(month, attribute = None, save = True)



