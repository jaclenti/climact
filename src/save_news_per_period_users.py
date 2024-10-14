import pandas as pd
import numpy as np
import sys
sys.path += ["../src"]
import climact_shared.src.utils as cu
from glob import glob
from functools import reduce
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import features_cox_week as ft





# def save_news_lag(news_df, lag, subreddit_class):


# average norm_* news in (country,admin1code) in a period of week_past weeks, ended week_lag weeks ago


def save_news_rolling_mean_df(week_lag = 1, weeks_past = 1, save = True):
    news_months_classes = pd.read_csv(cu.data_path + "news_count_us_canada_week.csv.gzip", 
                                      compression = "gzip", index_col = 0)

    news_weeks = [news_months_classes.query("is_" + news_class)
                  .groupby(["country", "region", "week_year"]).sum()["count"].reset_index()
                  .rename(columns = {"region": "admin1code", "count": "tot_" + news_class}) for news_class in ft.news_classes]    
    
    tot_news = (pd.concat([pd.read_csv(cu.data_path + f"news_count_us_canada/{d.strftime('%Y%m%d')}.csv.gz", compression = "gzip", index_col = 0)
                           .assign(day = d) for d in ft.all_dates[300:-1]])
                           .assign(week_year = lambda x: [ft.date_to_week_year(u) for u in x["day"]])
                           .groupby(["country", "admin1code", "week_year"])
                           .sum(numeric_only = True)
                           .reset_index()
                           .rename(columns = {"count": "tot_news"})
                           )
    
    news_weeks = reduce(lambda x, y: pd.merge(x,y), news_weeks + [tot_news])
    for news_class in ft.news_classes:
        news_weeks["norm_" + news_class] = news_weeks["tot_" + news_class] / news_weeks["tot_news"]
    
    rolling_mean_norm_news = (news_weeks
                              .groupby(["country", "admin1code"])
                              .rolling(window = weeks_past)
                              .mean(numeric_only = True)
                              [["norm_climate","norm_climate_action","norm_natural_disaster"]]
                              .reset_index()
                              .fillna(0)
                              .drop("level_2", axis = 1)
                              .assign(week_year = [ft.lag_week(week_year, lag = week_lag)
                                                   for week_year in news_weeks["week_year"]])
                                                   )
    if save:
        rolling_mean_norm_news.to_csv(cu.data_path + f"features_authors/rolling_mean_norm_news_weeklag{week_lag}_weekspast{weeks_past}.csv")
    
    return rolling_mean_norm_news


def save_news_geo_week_month_year(week_lags = [1,2,6], weekpasts = [1,4,46], filename = "news_features_geo_week_month_year"):
    news = {}
    news["short"] = pd.read_csv(cu.data_path + f"features_authors/rolling_mean_norm_news_weeklag{week_lags[0]}_weekspast{weekpasts[0]}.csv", index_col = 0)
    news["medium"] = pd.read_csv(cu.data_path + f"features_authors/rolling_mean_norm_news_weeklag{week_lags[1]}_weekspast{weekpasts[1]}.csv", index_col = 0)
    news["long"] = pd.read_csv(cu.data_path + f"features_authors/rolling_mean_norm_news_weeklag{week_lags[2]}_weekspast{weekpasts[2]}.csv", index_col = 0)
    news_features_geo = (reduce(lambda x,y: pd.merge(x,y, how = "outer", left_index = True, right_index = True), 
                                [ft.geo_users.merge(news[p]).set_index(["author", "week_year"])
                                 [["norm_climate", "norm_climate_action", "norm_natural_disaster"]]
                                 .add_suffix(f"_{p}") for p in news.keys()]))
    news_features_geo.to_csv(cu.data_path + f"features_authors/{filename}.csv.gz", compression = "gzip")

