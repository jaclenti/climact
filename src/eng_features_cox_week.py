import pandas as pd
import numpy as np
import sys
sys.path += ["../src"]
import climact_shared.src.utils as cu
from glob import glob
from functools import reduce
import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from time import time
import spark_init
pd.DataFrame.iteritems = pd.DataFrame.items
import pyspark.sql.functions as F
from pyspark.sql.functions import substring
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, LongType
sc = spark_init.spark_context()
sparkSQL = spark_init.SQLContext(sc)



schema = StructType([StructField("author", StringType(), True), 
                           StructField("item_id", StringType(), True),
                           StructField("week_year", StringType(), True)
                           ])

def load_histories_data(month, authors_df, type_txt = "RC"):
    path = cu.path_window + f"{type_txt}_{month}.parquet"
    if len(os.listdir(path)) == 0:
        return pd.DataFrame(columns = ["author", "week_year", "id", "subreddit"])
    else:
        return (sparkSQL.read.parquet(path)
                .filter(F.col("author").isin(authors_df.author.to_list()))
                .toPandas()
                .groupby("id").first().reset_index()
                .assign(date_comment = lambda x: cu.utc_to_date(x).dt.strftime("%Y-%m-%d"))
                .assign(week_year = lambda x: [f"{u.year}_{u.week:02d}" for u in pd.to_datetime(x["date_comment"])])
                )

def split_authors_weekly(df_window_month, authors_df, weeks_in_month):
    return (authors_df[authors_df.author.isin(set(df_window_month["author"]))]
            .assign(date_end = lambda x: pd.DataFrame({"strong": x["date_strong"] + pd.to_timedelta("7days"), 
                                                       "end": x["date_weak"] + pd.to_timedelta(f"{366 * 2}days")}).min(axis = 1),
                    date_start = lambda x: x["date_weak"] - pd.to_timedelta("366d"),
                    week_year_strong = lambda x: [cu.date_to_week_year(u) for u in pd.to_datetime(x["date_strong"])])
            .assign(weeks_range = lambda x: [list(pd.date_range(start = start, end = end, freq = "w"))
                                             for start, end in zip(x["date_start"],x["date_end"])])
            .explode("weeks_range")
            .assign(week_year = lambda x: [cu.date_to_week_year(u) for u in pd.to_datetime(x["weeks_range"])])
            .assign(activation = lambda x: x["week_year"] == x["week_year_strong"],
                    duration = lambda x: pd.DataFrame({"first_week": (x["weeks_range"] - x["date_start"]).dt.days, 
                                                       "full_week": 7, 
                                                       "activation_week":(x["date_end"] - x["weeks_range"]).dt.days})
                                                       .min(axis = 1))
            .query("week_year in @weeks_in_month")
            .reset_index().drop("index", axis = 1)
            .sort_values("weeks_range")
            .groupby(["author", "week_year"])
            .last()
            .reset_index()
            [["author", "week_year", "active", "date_weak", "date_strong", "weeks_range", "duration", "activation"]]
            )

def create_interaction_dataframe(month, df_window_month_rc, df_window_month_rs, subreddit_class, subreddit_list, interaction = ["parent_id", "parent_id"]):
    # interaction = [item_active, item_non_active]
    link_ids = np.load(glob(cu.data_path + f"{interaction[0]}_after_strong/RC_{month}.npy")[0], allow_pickle = True)
    if (interaction[1] == "id")&(link_ids.dtype == "O"):
        link_ids = np.array([u[3:] for u in link_ids])
    try:
        set_ids = set(link_ids) #if link_ids.dtype == "O" else set()
    except:
        print("no set_ids", month, interaction)
        set_ids = set()
    
    if interaction[1] == "id":
        df_window_month = pd.concat([df_window_month_rc[["author", "subreddit", "id", "week_year"]], 
                                     df_window_month_rs[["author", "subreddit", "id", "week_year"]]])
    else:
        df_window_month = df_window_month_rc.copy()
    
    if interaction[0] == "id":
        df_window_month[interaction[1]] = [u[3:] for u in df_window_month[interaction[1]]]

    intersection_item_id = list(set(df_window_month[interaction[1]])&set_ids)
    
    df_after_strong_intersection_com = (sparkSQL.read.parquet(cu.path_after + f"RC_{month}.parquet")
                                        .withColumn(interaction[0], substring(interaction[0], 4 * (interaction[1] == "id"), len = 20))
                                        .filter((F.col(interaction[0]).isin(intersection_item_id))
                                            &(~F.col("subreddit").isin(subreddit_list))
                                            &(F.col("subreddit_class") == subreddit_class))
                                            )
    
    df_window_intersection = (df_window_month
                              .rename(columns = {interaction[1]: "item_id"})[["author", "item_id", "week_year"]]
                              .query(f"item_id in @intersection_item_id"))
    # if (len(df_after_strong_intersection_com.head(1)) * len(df_window_intersection) == 0)&(interaction[0] != "id"):
        
    #     return pd.DataFrame(columns = ["item_id", "author", "week_year", "author_active", "subreddit"])
    if 2 == 3:
        0
    else:
        interactions_df_com = (sparkSQL.createDataFrame(df_window_intersection, schema = schema)
                                                    .join(df_after_strong_intersection_com
                                                        .withColumnRenamed(interaction[0], "item_id")
                                                    .select(["author", "item_id", "subreddit"])
                                                        .withColumnRenamed("author", "author_active"),
                                                        on = "item_id", how = "inner")
                                                        .filter(F.col("author") != F.col("author_active"))
                                                        )
        if interaction[0] != "id":
            interactions_df = interactions_df_com#.toPandas()
            print(interaction, interactions_df.count())

            return interactions_df
        else:
            try:
                df_after_strong_intersection_sub = (sparkSQL.read.parquet(cu.path_after + f"RS_{month}.parquet")
                                                    .withColumn(interaction[0], substring(interaction[0], 4 * (interaction[1] == "id"), len = 20))
                                                    .filter((F.col(interaction[0]).isin(intersection_item_id))
                                                            &(~F.col("subreddit").isin(subreddit_list))
                                                            &(F.col("subreddit_class") == subreddit_class)))
            
                interactions_df_sub = (sparkSQL.createDataFrame(df_window_intersection, schema = schema)
                                                                .join(df_after_strong_intersection_sub
                                                                    .withColumnRenamed(interaction[0], "item_id")
                                                                    .select(["author", "item_id", "subreddit"])
                                                                    .withColumnRenamed("author", "author_active"), 
                                                                    on = "item_id", how = "inner")
                                                                    .filter(F.col("author") != F.col("author_active"))
                                                                    )
                interactions_df = interactions_df_com.union(interactions_df_sub)
                # interactions_df = pd.concat([interactions_df_com.toPandas(), interactions_df_sub.toPandas()])
                print(interaction, interactions_df.count())
                return interactions_df
                
            except:
                print("df_after_strong_intersection_sub")
                return interactions_df_com#.toPandas()
        
    # interactions_df = (sparkSQL.createDataFrame(df_window_month.rename(columns = {interaction[1]: interaction[0]})[["author", interaction[0], "week_year"]])
    #                    .join(df_after_strong_intersection.select(["author", interaction[0], "subreddit"])
    #                          .withColumnRenamed("author", "author_active"), 
    #                          on = interaction[0], how = "inner")
    #                          .filter(F.col("author") != F.col("author_active"))
    #                          )
    



def lag_week(week_year, lag):
    index_week = cu.more_week_years.index(week_year)
    lagged_week = cu.more_week_years[index_week + lag]
    return lagged_week



if __name__ == "__main__":
    subreddit_class = sys.argv[1]
    subreddit_list = pd.read_pickle(cu.data_path + f"subreddit_list/{subreddit_class}_subreddits.pkl").to_list()
    
    df_strong = (pd.read_csv(cu.data_path + "author_list_subreddit_class/strong_authors_cohort_subreddit_class.csv", index_col = 0)
                 .assign(date_strong = lambda x: pd.to_datetime(x["date_strong"]), 
                         date_weak = lambda x: pd.to_datetime(x["date_weak"])).query("subreddit_class == @subreddit_class"))
    df_control = (pd.read_csv(cu.data_path + "author_list_subreddit_class/control_authors_cohort.csv", index_col = 0)
                  .sample(n = 3 * len(df_strong), random_state = 2006)
                  .assign(date_weak = lambda x: pd.to_datetime(x["date_weak"]), 
                          date_strong = pd.to_datetime("2100-01-01")))
    
    strong_activated_authors = pd.read_csv(cu.data_path + f"first_activation_subreddit_class/{subreddit_class}.csv", index_col = 0).author.to_list()
    
    authors_df = (pd.concat([(df_strong.assign(active = True).query("(subreddit_class == @subreddit_class)")), 
                             df_control.assign(active = False)])
                             [["author", "cohort", "active", "date_strong", "date_weak"]]
                             .sort_values("date_strong").groupby("author").first().reset_index())
    
    authors = authors_df.author.to_list()
    months_list = cu.all_months
    for month in months_list:
        print(subreddit_class, month)
        t0 = time()
        df_window_month_rc = load_histories_data(month, authors_df, "RC")
        df_window_month_rs = load_histories_data(month, authors_df, "RS")
        print(len(df_window_month_rc.author.unique()), "authors")
        weeks_in_month = df_window_month_rc.week_year.unique()
        author_week = split_authors_weekly(df_window_month_rc, authors_df, weeks_in_month)
        # control variables
        n_comments_author_week = (df_window_month_rc[["author", "week_year"]].value_counts().reset_index()
                                  .rename(columns = {"count": "n_comments_author_week"}))
        n_submissions_author_week = (df_window_month_rs[["author", "week_year"]].value_counts().reset_index()
                                  .rename(columns = {"count": "n_submissions_author_week"}))
        n_active_days_author_week = (pd.concat([df_window_month_rc[["author", "date_comment", "week_year"]]]).value_counts().reset_index()
                                     [["author", "week_year"]].value_counts().reset_index()
                                     .rename(columns = {"count": "n_active_days_author_week"}))
        avg_comments_per_thread = (df_window_month_rc[["author", "parent_id", "week_year"]].value_counts().reset_index()
                                   .groupby(["author", "week_year"]).mean(numeric_only = True)
                                   .reset_index().rename(columns = {"count": "avg_comments_per_thread"}))
        n_different_subreddits = (pd.concat([df_window_month_rc[["author", "week_year", "subreddit"]], df_window_month_rs[["author", "week_year", "subreddit"]]]).value_counts().reset_index()
                                  [["author", "week_year"]].value_counts().reset_index()
                                  .rename(columns = {"count": "n_different_subreddits"}))
        # interaction features
        # interactions_pd = create_interaction_dataframe(month, df_window_month, subreddit_class, subreddit_list)
        
        # n_different_comments_with_active = (interactions_pd[["author", "week_year"]].value_counts().reset_index()
        #                                     .rename(columns = {"count": "n_different_comments_with_active"}))
        # n_different_parent_id_with_active = (interactions_pd[["author", "parent_id", "week_year"]].value_counts().reset_index()
        #                                      [["author", "week_year"]].value_counts().reset_index()
        #                                      .rename(columns = {"count": "n_different_parent_id_with_active"}))
        # n_different_active_authors = (interactions_pd[["author", "author_active", "week_year"]].value_counts().reset_index()
        #                               [["author", "week_year"]].value_counts().reset_index()
        #                               .rename(columns = {"count": "n_different_active_authors"}))
        interactions_dataframes = []
        for interaction in [["parent_id", "parent_id"], ["link_id", "link_id"], ["parent_id", "id"], ["id", "parent_id"]]:
            interactions_sql = create_interaction_dataframe(month, df_window_month_rc, df_window_month_rs, subreddit_class, subreddit_list, interaction = interaction)
        
            n_different_comments_with_active = (interactions_sql.groupBy(["author", "week_year"]).count().toPandas().sort_values("count", ascending = False)
                                                .rename(columns = {"count": f"n_different_comments_with_active_{interaction[0]}_{interaction[1]}"}))
            # n_different_item_id_with_active = (interactions_pd[["author", "item_id", "week_year"]].value_counts().reset_index()
            #                                     [["author", "week_year"]].value_counts().reset_index()
            #                                     .rename(columns = {"count": f"n_different_parent_id_with_active_{interaction[0]}_{interaction[1]}"}))
            n_different_active_authors = (interactions_sql.groupBy(["author", "author_active", "week_year"]).count().toPandas().sort_values("count", ascending = False)
                                        [["author", "week_year"]].value_counts().reset_index()
                                        .rename(columns = {"count": f"n_different_active_authors_{interaction[0]}_{interaction[1]}"}))
            interactions_dataframes.append(n_different_comments_with_active)
            # interactions_dataframes.append(n_different_item_id_with_active)
            interactions_dataframes.append(n_different_active_authors)
        
        
        
        
        # all activity features
        # activity_features = reduce(lambda x, y: pd.merge(x,y, how = "left"), [author_week,
        #                                                                       n_different_comments_with_active,
        #                                                                       n_different_parent_id_with_active,
        #                                                                       n_different_active_authors,
        #                                                                       n_comments_author_week,
        #                                                                       n_active_days_author_week,
        #                                                                       avg_comments_per_thread,
                                                                            #   n_different_subreddits]).fillna(0)
        activity_features = reduce(lambda x, y: pd.merge(x,y, how = "left"), [author_week] + interactions_dataframes + 
                                                                              [n_comments_author_week,
                                                                               n_submissions_author_week,
                                                                               n_active_days_author_week,
                                                                               avg_comments_per_thread,
                                                                               n_different_subreddits]).fillna(0)
        # subreddit features
        # subreddits_features = get_subreddit_features(df_window_month, authors_df, author_week)
        authors_subreddits_month = (pd.concat([df_window_month_rc[["author", "subreddit", "week_year", "id"]], 
                                               df_window_month_rs[["author", "subreddit", "week_year", "id"]]])
                                    .merge(pd.DataFrame(cu.subreddits, # [u for u in subreddits if u not in subreddit_list], 
                                                        columns = ["subreddit"]), how = "right")
                                                        .fillna("-- NO  AUTHORS --"))
        # news
        # news_features = get_news_features(month, author_week)

        activity_features.to_csv(cu.data_path + f"features_authors/activity_features_{subreddit_class}_{month}.csv.gz", compression = "gzip")
        author_week.to_csv(cu.data_path + f"features_authors/author_week_{subreddit_class}_{month}.csv.gz", compression = "gzip")
        # news_features.to_csv(cu.data_path + f"features_authors/news_features_{subreddit_class}_{month}.csv.gz", compression = "gzip")
        authors_subreddits_month[["author", "week_year", "subreddit", "id"]].to_csv(cu.data_path + f"features_authors/authors_subreddits_month_{subreddit_class}_{month}.csv.gz",
                                                                                     compression = "gzip")
        t1 = time()
        print(round(t1 - t0, 1), "s")

        #     authors_subreddits_month = ((~authors_subreddits_month
        #                                  .groupby(["author", "week_year", "subreddit"]).count()["id"]
        #                                  .unstack()
        #                                  .drop("-- NO  AUTHORS --")
        #                                  .isna())
        #                                  .reset_index())
        #     subreddits_features = pd.merge(author_week, authors_subreddits_month, how = "left").fillna(False)

        # features_df = reduce(lambda x, y: pd.merge(x,y, how = "left"), [activity_features,
        #                                                                 subreddits_features,
        #                                                                 news_features])






        










    




















