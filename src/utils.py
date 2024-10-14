
import json
import pandas as pd
import numpy as np

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
txt = ["RS", "RC"]

subreddit_classes = ["activism", "action", "discussion", "skeptic"]
sociodemo_classes = ["age", "gender", "partisan", "affluence"]

def resilient_json(s):
    try:
        return json.loads(s)
    except:
        return {}
    


def compute_aic(N, LL, k):
    return - 2 / N * LL + 2 * k / N

def compute_bic(N, LL, k):
    return -2 * LL + np.log(N) * k



all_dates = pd.date_range(start = pd.to_datetime("2015-03-01"), end = pd.to_datetime("2023-01-01"))
all_months = [u.strftime("%Y-%m") for u in all_dates[all_dates.day == 1]]
all_week_years = (sorted(pd.Series([f"{u.year}_{u.week:02d}" for u in all_dates]).unique()))

more_dates = pd.date_range(start = pd.to_datetime("2012-03-01"), end = pd.to_datetime("2025-01-01"))
more_week_years = (sorted(pd.Series([f"{u.year}_{u.week:02d}" for u in more_dates]).unique()))

path_after = data_path + "histories_after_strong/"
path_window = data_path + "histories_strong_activated_and_control/"

news_classes = ["climate", "climate_action", "natural_disaster"]
geo_users = pd.read_csv(data_path + "geolocated_users_window.csv", index_col = 0)
subreddits = pd.read_csv(data_path + "subreddit_count_histories.csv.gzip", compression = "gzip", index_col = 0).head(1000).subreddit.to_list()

features = {"target":['activation', 'duration'],
            "control":['avg_comments_per_thread',
                       'n_submissions_author_week',
                       'n_active_days_author_week',
                       'n_comments_author_week',
                       'n_different_subreddits'],
            # "interaction":['n_different_active_authors',
            #                'n_different_comments_with_active',
            #                'n_different_parent_id_with_active'],
            "interaction": [u + "_" + i for u in ["n_different_comments_with_active", 
                                                #   "n_different_parent_id_with_active", 
                                                  "n_different_active_authors"] 
                            for i in ["parent_id_parent_id", 
                                      "link_id_link_id", 
                                      "link_id_id", 
                                      "id_link_id"]],
            "norm_news":['norm_climate',
                         'norm_climate_action',
                         'norm_natural_disaster'],
            "tot_news":['tot_climate',
                        'tot_climate_action',
                        'tot_natural_disaster',
                        'tot_news'],
            "subreddit": subreddits,
            "sociodemo": ["sociodemo_young", 
                        "sociodemo_old",
                        "sociodemo_male", 
                        "sociodemo_female",
                        "sociodemo_left",
                        "sociodemo_right",
                        "sociodemo_poor",
                        "sociodemo_rich"]
            # "sociodemo": [f'quartile{q}_{feature}' 
            #               for q in [1,4] 
            #               for feature in ["affluence", "age", "gender", "partisan"]]
            }
periods = ["short", "medium", "long", "short_long_ratio"]

features_time = {u: {p: [feat + "_" + p for feat in features[u]] for p in periods}
                 for u in ["control", "interaction", "norm_news", "subreddit"]}
features_time["subreddit"] = {p: ["r" + u for u in features_time["subreddit"][p]] for p in periods}

features_join = {"_".join([u,p]): features_time[u][p] for u in features_time for p in features_time[u]}
features_join["sociodemo"] = features["sociodemo"]


past_aggregation_intervals = {
    "week":
        {"past_days": 1, 
        "week_lag": 1,
        "normalized_variables": ["control", "interaction"]},
    "month":
        {"past_days": 28, 
        "week_lag": 2,
        "normalized_variables": ["control", "interaction", "norm_news", "tot_news"]},
    "year":
        {"past_days": 338, 
        "week_lag": 6,
        "normalized_variables": ["control", "interaction", "norm_news", "tot_news"]}
}

quartile_to_sociodemo = {"quartile1_age": "sociodemo_young", 
                        "quartile4_age": "sociodemo_old",
                        "quartile1_gender": "sociodemo_male", 
                        "quartile4_gender": "sociodemo_female",
                        "quartile1_partisan": "sociodemo_left",
                        "quartile4_partisan": "sociodemo_right",
                        "quartile1_affluence": "sociodemo_poor",
                        "quartile4_affluence": "sociodemo_rich"}

# parent_id_to_link_id = {"_".join([v,u,p]): "_".join([v,u,p]).replace("parent_id_id", "link_id_id").replace("id_parent_id", "id_link_id").replace("parent_id_link_id", "parent_id_parent_id") 
#                         for p in features_time["interaction"]
#                         for v in ["n_different_comments_with_active", "n_different_active_authors"]
#                         for u in ["parent_id_parent_id", 
#                                       "link_id_link_id", 
#                                       "parent_id_id", 
#                                       "id_parent_id"]
#                         }

parent_id_to_link_id = {"_".join([v,u,p]): "_".join([v,u,p]).replace("parent", "foo").replace("link", "parent").replace("foo", "link") 
                        for p in features_time["interaction"]
                        for v in ["n_different_comments_with_active", "n_different_active_authors"]
                        for u in ["parent_id_parent_id", 
                                      "link_id_link_id", 
                                      "parent_id_id", 
                                      "id_parent_id"]
                        }

selected_features = (features["sociodemo"] + 
                     features_time["interaction"]["short"] + 
                     features_time["control"]["short"] + 
                     features_time["control"]["long"] +
                     features_time["subreddit"]["long"] +
                     features_time["norm_news"]["long"] +
                     ["activation", "duration"])
def date_to_week_year(date):
    return f"{date.isocalendar().year}_{date.isocalendar().week:02d}"



def convert_pd_jnp_array(x):
    return np.array([u.item() for u in x])

def jnp_series_to_pd(x):
    try:
        return pd.Series([u.item() for u in x], index = x.index)
    except:
        return x
