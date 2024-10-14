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




def save_history_from_author_set(month, year, authors, type_txt, path):
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
    if not os.path.isdir(f"/data/big/xxx/climact/data/future_history_strong_authors/{type_txt}_{year}_{month}"):
        data_json.saveAsTextFile(f'/data/big/xxx/climact/data/future_history_strong_authors/{type_txt}_{year}_{month}', compressionCodecClass="org.apache.hadoop.io.compress.BZip2Codec")

if __name__ == '__main__':
    
    previous_activated_authors = []
    for u,month_year in enumerate(month_years):
        year, month = month_year.split("_")
        year_int, month_int = int(year), int(month)
        previous_activated_authors += list(strong_activations.query("(year == @year_int)&(month == @month_int)").author)
        print(month_year, len(previous_activated_authors))
        if len(previous_activated_authors) > 0:
            for type_txt, type_text in zip(type_txts, type_texts):
                path = f'/data/shared/reddit/{type_text}/{month_year.split("_")[0]}/{type_txt}_{month_year.split("_")[0]}-{month_year.split("_")[1]}.bz2'
                # print(path)
                if len(path) > 0:
                    save_history_from_author_set(month, year, previous_activated_authors, type_txt, path)
        

