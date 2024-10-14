import sys
import json
sys.path += ['../src/']
from time import time
from glob import glob
import pandas as pd
import climact_shared.src.utils as cu
import spark_init
sc = spark_init.spark_context()
import shutil



def resilient_json(s):
    try:
        return json.loads(s)
    except:
        return {}


def save_first_activation(data_path, subreddit_class):
    data = (sc.textFile(data_path)
            .map(cu.resilient_json)
            #  .map(lambda x: (x[0], x[1:]))
            #  .groupByKey()
            #  .map(lambda x: [x[0]] + list(x[1])[0])
            ).collect()
    df = (pd.DataFrame(data, columns = cu.columns)
          .assign(created_utc = lambda x: x["created_utc"].astype(int))
          .sort_values("created_utc")
          .groupby("author").first().reset_index()
          .assign(date = lambda x: [pd.to_datetime(int(u), unit = "s").strftime("%Y-%m-%d") 
                                    for u in x["created_utc"]])[["author", "subreddit", "date"]])
    
    df.to_csv(cu.data_path + f"first_activation_subreddit_class/{subreddit_class}.csv")
    


if __name__ == '__main__':
    subreddit_class = sys.argv[1]

    t0 = time()
    data_path = ",".join([u for year in range(2011, 2024) for u in sorted(glob(cu.data_path + f"subreddit_class_data/{subreddit_class}/*{year}*/*.bz2"))])
    save_first_activation(data_path, subreddit_class)
    t1 = time()
    print(t1 -t0, "s")

