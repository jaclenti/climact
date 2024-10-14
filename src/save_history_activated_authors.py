import numpy as np
import pandas as pd
import sys

sys.path += ['../src/']
from glob import glob
import sys
import json
sys.path += ['../src/']
import os
import spark_init
sc = spark_init.spark_context()

def resilient_json(s):
    try:
        return json.loads(s)
    except:
        return {}
    

data_path = "/data/big/xxx/climact/data/"

columns = ['author',
 'subreddit',
 'body',
 'created_utc',
 'id',
 'link_id',
 'score',
 'subreddit_id',
 'parent_id']


utc_to_date = lambda x: pd.to_datetime(x["created_utc"], unit = "s")

bot_list=list(pd.read_csv(data_path + 'bot_list.csv')['author'])

activation_types = ["strong", "weak"]
type_texts = ["submissions", "comments"]
type_txts = ["RS", "RC"]


### file  history_{activation}_authors/{type_txt}_{year}_{month} contain the post shared in month-year by users that will be activated

# def save_history_from_author_set(month, year, author_set, activation, type_txt):
#     data = (
#         # All of the following objects are pyspark.RDD, a (distributed) unordered set of objects (call them "lines").
#         # `sc.textFile(data_path)` reads the data from the path on the server and store them on a SparkContext object.
#         sc.textFile(data_path) # The read data are now stored in a RDD, where each line is a string.
#         .map(resilient_json) # Apply the function resilient_json to each row: each line become a dictionary.
#         .map(lambda x: (x.get("author"),
#                 x.get("subreddit"),
#                 x.get("body") if type_txt == "RC" else x.get("selftext"),
#                 x.get("created_utc"),
#                 x.get("id"),
#                 x.get("link_id"),
#                 x.get("score"),
#                 x.get("subreddit_id"),
#                 x.get("parent_id")))
#         .filter(lambda x: x[0] in author_set) # fatto prima del json
#     )

#     # data.cache()

#     data_json = data.map(lambda x: json.dumps(x))
#     if not os.path.isdir(f"/data/big/xxx/climact/data/history_{activation}_authors/{type_txt}_{year}_{month}"):
#         data_json.saveAsTextFile(f'/data/big/xxx/climact/data/history_{activation}_authors/{type_txt}_{year}_{month}', compressionCodecClass="org.apache.hadoop.io.compress.BZip2Codec")

# if __name__ == '__main__':
#     activation = sys.argv[1]

#     authors = pd.concat([pd.read_pickle(file) for file in sorted(glob(data_path + f"/first_activations/{activation}_first_activations_*.pkl"))])    
#     author_month_year_sets = {f"{year}_{month:02d}": set(authors.query("(year == @year)&(month == @month)")["author"]) for year in range(2011, 2023) for month in range(1,13)}
    
#     for u, month_year in enumerate(author_month_year_sets.keys()):
#         months_previous_2years = list(author_month_year_sets.keys())[max(0, u - 24):u]
#         year, month = month_year.split("_")
#         author_set = [authors for month_year in months_previous_2years for authors in author_month_year_sets[month_year]]
#         print(year, month, len(author_set))
#         if len(author_set) > 0:
#             data_path_RC = f'/data/shared/reddit/comments/{year}/RC_{year}-{month}.*'
#             data_path_RS = f'/data/shared/reddit/submissions/{year}/RS_{year}-{month}.*'
#             for type_txt, data_path in zip(["RC", "RS"], [data_path_RC, data_path_RS]):
#                 save_history_from_author_set(month, year, author_set, activation, type_txt)


### file  history_{activation}_authors/{type_txt}_{year}_{month} contain all posts of users activated in month-year


def save_history_from_author_set(month, year, authors, activation, type_txt, path):
    data = (
        # All of the following objects are pyspark.RDD, a (distributed) unordered set of objects (call them "lines").
        # `sc.textFile(data_path)` reads the data from the path on the server and store them on a SparkContext object.
        sc.textFile(path) # The read data are now stored in a RDD, where each line is a string.
        .map(resilient_json) # Apply the function resilient_json to each row: each line become a dictionary.
        .map(lambda x: (x.get("author"),
                x.get("subreddit"),
                x.get("body") if type_txt == "RC" else x.get("selftext"),
                x.get("created_utc"),
                x.get("id"),
                x.get("link_id"),
                x.get("score"),
                x.get("subreddit_id"),
                x.get("parent_id")))
        .filter(lambda x: x[0] in authors) # fatto prima del json
    )

    # data.cache()

    data_json = data.map(lambda x: json.dumps(x))
    if not os.path.isdir(f"/data/big/xxx/climact/data/history_{activation}_authors/{type_txt}_{year}_{month}"):
        data_json.saveAsTextFile(f'/data/big/xxx/climact/data/history_{activation}_authors/{type_txt}_{year}_{month}', compressionCodecClass="org.apache.hadoop.io.compress.BZip2Codec")

if __name__ == '__main__':
    
    activation = sys.argv[1]
    month_years = [f"{year}_{month:02d}" for year in range(2011, 2023) for month in range(1,13)]
    for u,month_year in enumerate(month_years):
        year, month = month_year.split("_")
        previous_month_years = month_years[max(0, u - 24):u]
        authors = list(pd.read_pickle(data_path + f"/first_activations/{activation}_first_activations_{year}_{month}.pkl").reset_index()["author"])
        print(month_year, len(authors))
        if len(authors) > 0:
            for type_txt, type_text in zip(type_txts, type_texts):
                path = ",".join([f'/data/shared/reddit/{type_text}/{month_year.split("_")[0]}/{type_txt}_{month_year.split("_")[0]}-{month_year.split("_")[1]}.bz2' 
                                    for month_year in previous_month_years])
                if len(path) > 0:
                    save_history_from_author_set(month, year, authors, activation, type_txt, path)

