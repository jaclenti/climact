import sys
import json
import pandas as pd
sys.path += ['../src/']
from glob import glob
import climact_shared.src.utils as cu
import numpy as np
from time import time
import os

import spark_init
sc = spark_init.spark_context()

import pyspark.sql.functions as F

def resilient_json(s):
    try:
        return json.loads(s)
    except:
        return {}
    
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, LongType
spark = SparkSession(sc)
from pyspark.sql.functions import col



pd.DataFrame.iteritems = pd.DataFrame.items


schema = {"RC":StructType([StructField("author", StringType(), True), 
                           StructField("subreddit", StringType(), True),
                           StructField("body", StringType(), True),
                           StructField("created_utc", LongType(), True), 
                           StructField("id", StringType(), True),
                           StructField("link_id", StringType(), True),
                           StructField("score", LongType(), True),
                           StructField("subreddit_id", StringType(), True),
                           StructField("parent_id", StringType(), True),
                           ]),
         "RS": StructType([StructField("author", StringType(), True), 
                           StructField("subreddit", StringType(), True),
                           StructField("selftext", StringType(), True),
                           StructField("created_utc", LongType(), True), 
                           StructField("id", StringType(), True),
                           StructField("score", LongType(), True),
                           StructField("subreddit_id", StringType(), True),
                           ])}

get_features = {"RS": lambda x: (x.get("author"),
                                 x.get("subreddit"),
                                x.get("selftext"),
                                x.get("created_utc"),
                                x.get("id"),
                                x.get("score"),
                                x.get("subreddit_id"),
                                ),
                 "RC": lambda x: (x.get("author"),
                                x.get("subreddit"),
                                x.get("body"),
                                x.get("created_utc"),
                                x.get("id"),
                                x.get("link_id"),
                                x.get("score"),
                                x.get("subreddit_id"),
                                x.get("parent_id"),
                                ),
                                }


if __name__ == '__main__':
    all_strong = {subreddit_class: pd.read_csv(cu.data_path + f"first_activation_subreddit_class/{subreddit_class}.csv", index_col = 0)
                                         for subreddit_class in cu.subreddit_classes}
    all_strong_df = pd.concat(all_strong).reset_index().rename(columns = {"level_0": "subreddit_class"}).drop(["level_1", "subreddit"], axis = 1)
    all_months = sorted(pd.to_datetime(all_strong_df[pd.to_datetime(all_strong_df["date"]) > pd.to_datetime("2021-04-01")].date).dt.strftime("%Y-%m").unique())
    # all_months = sorted(pd.to_datetime(all_strong_df[pd.to_datetime(all_strong_df["date"]) > pd.to_datetime("2014-12-31")].date).dt.strftime("%Y-%m").unique())
    
    for k,month in enumerate(all_months):
        next_month = all_months[k+1]
        
        activated_authors_df = spark.createDataFrame(all_strong_df[pd.to_datetime(all_strong_df["date"]) < pd.to_datetime(next_month)])
        activated_authors = all_strong_df[pd.to_datetime(all_strong_df["date"]) < pd.to_datetime(next_month)].author.to_list()
        print(month, activated_authors_df.count())
        
        for type_txt, type_text in zip(["RC"], ["comments"]):
        # for type_txt, type_text in zip(["RS", "RC"], ["submissions", "comments"]):
            t0 = time()
            print(type_txt)
            reddit_path = f"/data/shared/reddit/{type_text}/{month.split('-')[0]}/{type_txt}_{month}.bz2"
            new_path = cu.data_path + f"histories_after_strong/{type_txt}_{month}.parquet"
            cols = {"RC": cu.columns, "RS": ["author","subreddit","selftext","created_utc","id","score","subreddit_id"]}
            
            ### RDD + SparkSQL
            
            if not os.path.exists(new_path):
                (sc.textFile(reddit_path)
                .map(resilient_json) 
                .map(get_features[type_txt])
                .filter(lambda x: x[0] in activated_authors)
                .toDF(schema = schema[type_txt])
                .join(F.broadcast(activated_authors_df),"author", "inner")
                .write.format('parquet')
                .save(cu.data_path + f"histories_after_strong/{type_txt}_{month}.parquet"))
                
                t1 = time()
                print(round((t1 - t0)/3600, 3), "hr")
            
            
            # try:
                
                
            #     df = (spark.read.json(reddit_path)
            #         .select(cols[type_txt])
            #         .filter(F.col("author").isin(activated_authors))
            #         # .join(F.broadcast(activated_authors_df),"author", "inner")
            #     )
                    
            #     df.write.format('parquet').save(cu.data_path + f"histories_after_strong/{type_txt}_{month}.parquet")

            #     t1 = time()
            #     print(round((t1 - t0)/3600, 3), "hr")
            # except:
            #     print("ERROR")
                
                

                

