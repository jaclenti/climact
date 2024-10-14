from glob import glob
import sys
sys.path += ["../src"]
import pandas as pd
import numpy as np
import features_cox_week as ft
import climact_shared.src.utils as cu
from functools import reduce
import warnings
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
warnings.simplefilter(action='ignore', category=pd.errors.SettingWithCopyWarning)
# warnings.simplefilter(action='ignore', category=pd.errors.SettingWithCopyWarning)

def previous_period_author_subreddit(sample_df, authors_subreddits_month, past_short_start, past_short_end, suffix):
    return (sample_df
            .reset_index()
            [["author", "week_year"]]
            .assign(previous_month = lambda x: [[ft.lag_week(w, -u) for u in range(past_short_end,past_short_start)] for w in x["week_year"]])
            .explode("previous_month")
            .merge(authors_subreddits_month[["author", "subreddit", "week_year"]]
                   .drop_duplicates()
                   .assign(write = True)
                   .rename(columns = {"week_year": "previous_month"}))
            .drop("previous_month", axis = 1)
            .drop_duplicates()
            .set_index(["author", "week_year", "subreddit"])
            .unstack()["write"]
            .fillna(False)
            .add_suffix(suffix)
            .add_prefix("r")
            )

def previous_period_author_activity(sample_df, activity_features, past_week_start, past_week_end, suffix):
    return (sample_df
            .reset_index()
            [["author", "week_year"]]
            .assign(previous_month = lambda x: [[ft.lag_week(w, -u) for u in range(past_week_end,past_week_start)] for w in x["week_year"]])
            .explode("previous_month")
            .merge(activity_features
                   .reset_index()
                   .rename(columns = {"week_year": "previous_month"}))
            .drop("previous_month", axis = 1)
            .drop_duplicates()
            .groupby(["author", "week_year"])
            .mean()
            .add_suffix(suffix)
            )


def sample_authors_from_activity(subreddit_class, months_list = cu.all_months, random_state = 1, balanced_classes = True):
    author_week = (pd.concat([pd.read_csv(cu.data_path + f"features_authors/author_week_{subreddit_class}_{month}.csv.gz", 
                                          compression = "gzip", index_col = 0) 
                                          for month in months_list]).groupby(["author", "week_year"]).first())
    
    activity_features = (pd.concat([pd.read_csv(cu.data_path + f"features_authors/activity_features_{subreddit_class}_{month}.csv.gz", 
                                                compression = "gzip", index_col = 0) 
                                                for month in months_list]).drop(["duration", "active", "activation"], axis = 1)
                                                .groupby(["author", "week_year"]).sum(numeric_only = True))
    if balanced_classes:
        sample_negative = 2 * len(author_week.query("activation"))

    sample_df = pd.concat([author_week
                           .query("(activation)"), 
                           author_week.assign(date_weak = lambda x: pd.to_datetime(x["date_weak"]),
                                              date_strong = lambda x: pd.to_datetime(x["date_strong"]),
                                              weeks_range = lambda x: pd.to_datetime(x["weeks_range"]))
                                              .query("(~activation)&(date_weak < weeks_range <= date_strong)")
                                              .sample(sample_negative, replace = False, random_state = random_state)])
    
    sample_df.to_csv(cu.data_path + f"sample_authors/{subreddit_class}_sample.csv")
    


def create_features_df(subreddit_class, months_list = cu.all_months, random_state = 1, 
                       short_start_end = [2,1], medium_start_end = [6,2], long_start_end = [53,6],
                       sample_negative = 10000, balanced_classes = False, verbose = False):
    short_start, short_end = sorted(short_start_end, reverse = True)
    medium_start, medium_end = sorted(medium_start_end, reverse = True)
    long_start, long_end = sorted(long_start_end, reverse = True)
    
        
    
    if verbose:
        print("Activity features")
    author_week = (pd.concat([pd.read_csv(cu.data_path + f"features_authors/author_week_{subreddit_class}_{month}.csv.gz", 
                                          compression = "gzip", index_col = 0) 
                                          for month in months_list]).groupby(["author", "week_year"]).first())
    if balanced_classes:
        sample_negative = 2 * len(author_week.query("activation"))

    
    activity_features = (pd.concat([pd.read_csv(cu.data_path + f"features_authors/activity_features_{subreddit_class}_{month}.csv.gz", 
                                                compression = "gzip", index_col = 0) 
                                                for month in months_list]).drop(["duration", "active", "activation"], axis = 1)
                                                .groupby(["author", "week_year"]).sum(numeric_only = True))
    
    # sample_df = pd.concat([author_week
    #                        .query("(activation)"), 
    #                        author_week.assign(date_weak = lambda x: pd.to_datetime(x["date_weak"]),
    #                                           date_strong = lambda x: pd.to_datetime(x["date_strong"]),
    #                                           weeks_range = lambda x: pd.to_datetime(x["weeks_range"]))
    #                                           .query("(~activation)&(date_weak < weeks_range <= date_strong)")
    #                                           .sample(sample_negative, replace = False, random_state = random_state)])
    sample_df = pd.read_csv(cu.data_path + f"sample_authors/{subreddit_class}_sample.csv", index_col = [0,1])
    previous_short_activity = previous_period_author_activity(sample_df, activity_features, short_start, short_end, "_short")
    previous_medium_activity = previous_period_author_activity(sample_df, activity_features, medium_start, medium_end, "_medium")
    previous_long_activity = previous_period_author_activity(sample_df, activity_features, long_start, long_end, "_long")
    if verbose:
        print("Subreddit features")
    authors_subreddits_month = pd.concat([pd.read_csv(cu.data_path + f"features_authors/authors_subreddits_month_{subreddit_class}_{month}.csv.gz", 
                                                      compression = "gzip", lineterminator = "\n", 
                                                      low_memory = False, index_col = 0)[["author", "subreddit", "week_year"]] 
                                                      for month in months_list])

    previous_short_subreddit = previous_period_author_subreddit(sample_df, authors_subreddits_month, short_start, short_end, "_short")
    previous_medium_subreddit = previous_period_author_subreddit(sample_df, authors_subreddits_month, medium_start, medium_end, "_medium")
    previous_long_subreddit = previous_period_author_subreddit(sample_df, authors_subreddits_month, long_start, long_end, "_long")
    if verbose:
        print("News features")
    news_features_geo = pd.read_csv(cu.data_path + "features_authors/news_features_geo_week_month_year.csv.gz", 
                                    compression = "gzip", index_col = [0,1])

    medians_news = news_features_geo.unstack().median().unstack().T
    
    sample_features_df = sample_df[["activation", "duration"]].merge(news_features_geo, how = "left", left_index = True, right_index = True)
    
    sample_features_df = pd.concat([sample_features_df.query("norm_climate_short == norm_climate_short"),
                                    sample_features_df.query("norm_climate_short != norm_climate_short")[["activation", "duration"]]
                                    .reset_index()
                                    .merge(medians_news.reset_index())
                                    .set_index(["author", "week_year"])])
    
    
    # sample_features_df.fillna(sample_features_df.median(), inplace = True)
    if verbose:
        print("Merge")
    features_df = reduce(lambda x,y: pd.merge(x,y, left_index = True, right_index = True, how = "left"), 
                        [sample_features_df,
                         previous_short_activity,
                         previous_medium_activity,
                         previous_long_activity]).fillna(0)
    
    features_df = reduce(lambda x,y: pd.merge(x,y, left_index = True, right_index = True, how = "left"), 
                        [features_df,
                         previous_short_subreddit, 
                         previous_medium_subreddit, 
                         previous_long_subreddit]).fillna(False)
    
                        
    features_df = features_df.query("(norm_natural_disaster_long > 0)&(norm_climate_long > 0)&(norm_climate_action_long > 0)")
    if verbose:
        print("Sociodemographic features")
    # for file in sorted(glob(cu.data_path + "sociodemographic_features/*1*npy") + glob(cu.data_path + "sociodemographic_features/*4*npy")):
    #     col_name = "_".join(file.split("/")[-1].split("_")[:-1])
    for file in sorted(glob(cu.data_path + f"sociodemographic_features/*1*{subreddit_class}*npy") + glob(cu.data_path + f"sociodemographic_features/*4*{subreddit_class}*npy")):
        col_name = "_".join(file.split("/")[-1].split("_")[:-2])
        features_df.loc[:,col_name] = features_df.copy().reset_index()["author"].isin(np.load(file)).tolist()
    features_df.rename(columns = cu.quartile_to_sociodemo, inplace = True)
    
    for feature in (cu.features["norm_news"]+cu.features["interaction"]+cu.features["control"]):
        # features_df.loc[:,feature + "_short_long_ratio"] = features_df.copy()[feature + "_short"] / features_df.copy()[feature + "_long"]
        features_df.loc[:,feature + "_short_long_ratio"] =  features_df[feature + "_short"].div(features_df[feature + "_long"], fill_value = 0).replace([np.inf, -np.inf], 1).fillna(2)
    
    subreddits_short_long = list(set([subreddit for subreddit in cu.features["subreddit"] if "r" + subreddit + "_short" in features_df.columns])&set([subreddit for subreddit in cu.features["subreddit"] if "r" + subreddit + "_long" in features_df.columns]))
    
    features_df.loc[:, ["r" + subreddit + "_short_long_ratio" for subreddit in subreddits_short_long]] = (np.array(features_df.loc[:, ["r" + subreddit + "_short" for subreddit in subreddits_short_long]] + 0.) - np.array(features_df.loc[:, ["r" + subreddit + "_long" for subreddit in subreddits_short_long]] + 0.)) > 0
    # for feature in cu.features["subreddit"]:
    #     if ("r" + feature + "_short" in features_df.columns)&("r" + feature + "_long" in features_df.columns):
    #         features_df.loc[:,"r" + feature + "_short_long_ratio"] = ((features_df.copy()["r" + feature + "_short"] + 0.) - (features_df.copy()["r" + feature + "_long"] + 0.)) > 0
        
    features_df.drop_duplicates(inplace = True)
    positive_authors = len(features_df.query("activation"))
    if balanced_classes:
        features_df = pd.concat([features_df.query("activation"), 
                                 features_df.query("~activation").sample(n = positive_authors, random_state = random_state)])
    features_df.rename(columns = cu.quartile_to_sociodemo, inplace = True)
    features_df.drop([u for u in features_df.columns if "n_different_parent_id" in u], axis = 1, inplace = True)
    
    return features_df

