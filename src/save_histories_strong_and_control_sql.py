import sys
import json
import pandas as pd
sys.path += ['../src/']
from glob import glob
import climact_shared.src.utils as cu
import numpy as np
from time import time


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
   
def cohort_to_dates(cohort):
    year, quarter = cohort.split("-")
    month = (int(quarter) - 1) * 3 + 1
    start_year = str(int(year)-1)
    start_month = str(month)
    end_year = str(int(year)+2)
    end_month = str((month + 3) % 12)
    start_date = pd.to_datetime(f"{start_year}-{start_month}-1")
    end_date = pd.to_datetime(f"{end_year}-{end_month}-1") - pd.to_timedelta("1d")
    return pd.to_datetime(start_date), pd.to_datetime(end_date)


if __name__ == '__main__':
    df_strong = pd.read_csv(cu.data_path + "strong_authors_cohort_subreddit_class.csv", index_col = 0)
    df_control = pd.read_csv(cu.data_path + "author_list_subreddit_class/control_authors_cohort.csv", index_col = 0)
    df_authors = pd.concat([df_control, df_strong])

    cohort_date_interval = {u: cohort_to_dates(u) for u in sorted(df_strong.cohort.unique())}
    all_dates = pd.date_range(start = cohort_date_interval["2015-1"][0], end = cohort_date_interval["2022-2"][1])
    all_months = [u.strftime("%Y-%m") for u in all_dates[all_dates.day == 1]]
    
    months_per_cohort = {cohort: all_months[(3*k):(3*k+39)] for k, cohort in enumerate(sorted(cohort_date_interval.keys()))}
    cohorts_per_month = {month: [cohort for cohort in months_per_cohort.keys() if month in months_per_cohort[cohort]] 
                     for month in all_months}
    for month in all_months[91:]:
        authors_in_window = df_authors[df_authors["cohort"].isin(cohorts_per_month[month])].author.to_list()
        authors_in_window_df = spark.createDataFrame([tuple([u]) for u in authors_in_window],
                                                     schema = StructType([StructField("author", StringType(), True)]))

        print(month, len(authors_in_window))
        for type_txt, type_text in zip(["RC"], ["comments"]):
        # for type_txt, type_text in zip(["RS", "RC"], ["submissions", "comments"]):
            t0 = time()
            print(type_txt)
            reddit_path = f"/data/shared/reddit/{type_text}/{month.split('-')[0]}/{type_txt}_{month}.bz2"

            cols = {"RC": cu.columns, "RS": ["author","subreddit","selftext","created_utc","id","score","subreddit_id"]}
            ### RDD + SparkSQL
            (sc.textFile(reddit_path)
            .map(resilient_json) 
            .map(get_features[type_txt])
            .filter(lambda x: x[0] in authors_in_window)
            .toDF(schema = schema[type_txt])
            .join(F.broadcast(authors_in_window_df),"author", "inner")
            .write.format('parquet')
            .save(cu.data_path + f"histories_strong_activated_and_control/{type_txt}_{month}.parquet"))
            
            t1 = time()
            print(round((t1 - t0)/3600, 3), "hr")


            t1 = time()
            print(round((t1 - t0)/3600, 3), "hr")
            #
            # 
            #  try:
            #     ### RDD + SparkSQL
            #     (sc.textFile(reddit_path)
            #      .map(resilient_json) 
            #      .map(lambda x: (x.get("author"),
            #                      x.get("subreddit"),
            #                      x.get("body") if type_txt == "RC" else x.get("selftext"),
            #                      x.get("created_utc"),
            #                      x.get("id"),
            #                      x.get("link_id"),
            #                      x.get("score"),
            #                      x.get("subreddit_id"),
            #                      x.get("parent_id")))
            #      .filter(lambda x: x[0] in authors_in_window)
            #      .toDF(schema = schema)
            #      .join(F.broadcast(authors_in_window_df),"author", "inner")
            #      .write.format('parquet')
            #      .save(cu.data_path + f"histories_strong_activated_and_control/{type_txt}_{month}.parquet"))


                                  



            #     ### SparkSQL
            #     # (spark.read.json(reddit_path)
            #     # .select(cols[type_txt])
            #     # .filter(F.col("author").isin(authors_in_window))
            #     # # .join(F.broadcast(authors_in_window_df),"author", "inner")
            #     # .write.format('parquet')
            #     # .save(cu.data_path + f"histories_strong_activated_and_control/{type_txt}_{month}.parquet"))

            #     t1 = time()
            #     print(round((t1 - t0)/3600, 3), "hr")
            # except:
            #     print("ERROR")                
                

                

