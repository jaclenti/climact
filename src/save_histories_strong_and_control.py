import sys
import json
import pandas as pd
sys.path += ['../src/']
from glob import glob
import climact_shared.src.utils as cu
import numpy as np
import polars as pl
import bz2
from time import time

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
    df_strong = pd.read_csv(cu.data_path + "author_list_subreddit_class/strong_authors_cohort_subreddit_class.csv", index_col = 0)
    df_control = pd.read_csv(cu.data_path + "author_list_subreddit_class/control_authors_cohort.csv", index_col = 0)
    df_authors = pd.concat([df_control, df_strong])

    cohort_date_interval = {u: cohort_to_dates(u) for u in sorted(df_strong.cohort.unique())}
    all_dates = pd.date_range(start = cohort_date_interval["2015-1"][0], end = cohort_date_interval["2022-2"][1])
    all_months = [u.strftime("%Y-%m") for u in all_dates[all_dates.day == 1]]
    
    months_per_cohort = {cohort: all_months[(3*k):(3*k+39)] for k, cohort in enumerate(sorted(cohort_date_interval.keys()))}
    cohorts_per_month = {month: [cohort for cohort in months_per_cohort.keys() if month in months_per_cohort[cohort]] 
                     for month in all_months}
    for month in all_months:
        authors_in_window = df_authors[df_authors["cohort"].isin(cohorts_per_month[month])].author.to_list()
        
        print(month, len(authors_in_window))
        for type_txt, type_text in zip(["RC", "RS"], ["comments", "submissions"]):
            t0 = time()
            print(type_txt)
            reddit_path = f"/data/shared/reddit/{type_text}/{month.split('-')[0]}/{type_txt}_{month}.bz2"
            df = (pl.read_ndjson(bz2.BZ2File(reddit_path).read())
                  .select(cu.columns)
                  .filter(pl.col("author").is_in(authors_in_window)))
            df.write_parquet(cu.data_path + f"histories_strong_activated_and_control/{type_txt}_{month}.gzip", compression="gzip")
            t1 = time()
            print(round((t1 - t0)/3600, 3), "hr")
            
            

            

