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


# Method for "unpack" the tuple and write the useful keys values as strings separated by commas:
def rdd_line_to_string(x):
#    (i, a, s, c) = x # Unpack the tuple
    (i, a, s, c) = x # Unpack the tuple
#    return str(i) + ',' + a + ',' + s + ',' + str(c)
    return str(i) + '|' + str(a) + '|' + str(s) + '|' + str(c)


def resilient_json(s):
    try:
        return json.loads(s)
    except:
        return {}


subreddit_classes = ["discussion", "action", "activism", "skeptic"]

def save_data_subreddits(data_path, subreddit_list, subreddit_class, year, month, type_txt):
    # data_path = f'/data/shared/reddit/{type_text}/{year}/{type_txt}_{year}-{month:02d}.bz2'
    
    data = (
        # All of the following objects are pyspark.RDD, a (distributed) unordered set of objects (call them "lines").
        # `sc.textFile(data_path)` reads the data from the path on the server and store them on a SparkContext object.
        sc.textFile(data_path) # The read data are now stored in a RDD, where each line is a string.
        .map(resilient_json) # Apply the function resilient_json to each row: each line become a dictionary.
        .map(lambda x: (x.get("author"),
                        x.get("subreddit"),
                        x.get("body") if type_txt == "RC" else x.get("selftext"),
                        x.get("created_utc"),
                        x.get("id"),
                        x.get("link_id"),
                        x.get("score"),
                        x.get("subreddit_id"),
                        x.get("parent_id"))
                )
        .filter(lambda x: x[1] in subreddit_list) 
        .map(lambda x: json.dumps(x))
        ).saveAsTextFile(cu.data_path + f'subreddit_class_data/{subreddit_class}/{type_txt}_{year}_{month:02d}', 
                             compressionCodecClass="org.apache.hadoop.io.compress.BZip2Codec")




if __name__ == '__main__':
    subreddit_class = sys.argv[1]
    subreddit_list = list(pd.read_pickle(cu.data_path + f"subreddit_list/{subreddit_class}_subreddits.pkl"))
    
    
    years = [y for y in range(2011, 2015)]
    # years = [y for y in range(2015, 2024)]
    months = [m for m in range(1, 13)]
    
    for year in years:
        for month in months:
            for type_txt, type_text in zip(["RS", "RC"], ["submissions", "comments"]):
                t0 = time()
                data_path = f"/data/shared/reddit/{type_text}/{year}/{type_txt}_{year}-{month:02d}.bz2"
                save_data_subreddits(data_path, subreddit_list, subreddit_class, year, month, type_txt)
                    
                t1 = time()
                print(subreddit_class, year, month, f"{round((t1-t0)/3600, 3)}hr")
        



