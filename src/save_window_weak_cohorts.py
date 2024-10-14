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
from time import time
from bz2 import BZ2File as bzopen


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
    t0 = time()
    
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

    data_json = data.map(lambda x: json.dumps(x))
    
    if not os.path.isdir(f"/data/big/xxx/climact/data/window_weak_activation_cohorts/{type_txt}_{year}_{month}"):
        data_json.saveAsTextFile(f'/data/big/xxx/climact/data/window_weak_activation_cohorts/{type_txt}_{year}_{month}', 
                                 compressionCodecClass="org.apache.hadoop.io.compress.BZip2Codec")
    

    t1 = time()
    print(round((t1 - t0) / 3600, 3), "hours")



if __name__ == '__main__':
     year0, year1 = sys.argv[1], sys.argv[2]
     month_years = [f"{year}_{month:02d}" for year in range(2016, 2024) for month in range(1,13)]
     weak_activations = pd.read_csv(data_path+"author_weak_activation_date.csv").assign(month_year = lambda x: [f"{y}_{m:02d}" for m,y in zip(x["month"], x["year"])])
     month_cohorts = [month_years[(3*k):3*(k+1)] for k in range(30)]
     
     for cohort in month_cohorts:
          print(cohort[0])
          year, month = cohort[0].split("_")
          if (year in [year0, year1])&(cohort[0] != "2016_01"):
            window_month_year = {month_cohorts[k][0]: month_cohorts[max(0,k-8):min(28, k+5)]  for k in range(30)}[cohort[0]]
            
            months_in_window = [u for m in window_month_year for u in m]
            authors_in_window = list(weak_activations.query("month_year in @months_in_window").author)
            print(len(authors_in_window))
            for type_txt, type_text in zip(type_txts, type_texts):
                
                print(type_txt)
                path = ",".join([f'/data/shared/reddit/{type_text}/{month_year.split("_")[0]}/{type_txt}_{month_year.split("_")[0]}-{month_year.split("_")[1]}.bz2' 
                                    for month_year in cohort])
                
                save_history_from_author_set(month, year, authors_in_window, type_txt, path)
                




