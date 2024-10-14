import pandas as pd
import lifelines
import sys
sys.path += ["../src"]
import features_cox_week as ft
import climact_shared.src.utils as cu
import numpy as np
from importlib import reload
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from time import time
import statsmodels.api as sm
import statsmodels.formula.api as smf

import sys

def get_features_authors(month_start, month_end, subreddit_class, save = True, return_df = False):
    months_list = ft.all_months[ft.all_months.index(month_start):(ft.all_months.index(month_end) + 1)]
    author_week = pd.concat([pd.read_csv(cu.data_path + f"features_authors/author_week_{subreddit_class}_{month}.csv.gz", 
                                         compression = "gzip", index_col = 0) for month in months_list]).groupby(["author", "week_year"]).first()
    print("Authors", len(author_week))
    print("Active authors", author_week.activation_week.sum())
    t0 = time()

    if save:
        ft.get_aggregate_features_from_months(months_list, subreddit_class).to_parquet(cu.data_path + f"features_authors/{subreddit_class}_{months_list[0]}_{months_list[1]}.parquet", compression = "gzip")
        t1 = time()
        print(t1 -t0)
    if return_df:
        df = ft.get_aggregate_features_from_months(months_list, subreddit_class)
        t1 = time()
        print(t1 -t0)
        return df


def get_sample_df(df, random_state = 1, positive_size = None, negative_size = 1000):
    df_positive = df.query("activation_week_week")
    df_positive = df_positive.drop_duplicates()
    if positive_size is not None:
        df_positive = df_positive.sample(positive_size, random_state = random_state)
    df_negative = df.query("~activation_week_week").sample(negative_size + 200, random_state = random_state)
    df_negative.drop_duplicates(inplace = True)
    df_negative = df_negative.sample(negative_size, random_state = random_state)
    df_sample = pd.concat([df_positive, df_negative])
    
    ############
    for term in ["_week", "_month", "_year"]:
        df_sample["avg_comments_per_thread" + term] = df_sample["avg_comments_per_thread" + term] * df_sample["duration" + term]
    ############
    df_sample.drop(['active_week','tot_climate_week','tot_climate_action_week','tot_natural_disaster_week',
 'tot_news_week','activation_week_month', 'active_week','tot_climate_week','tot_climate_action_week','tot_natural_disaster_week',
 'tot_news_week','activation_week_year', 'active_week','tot_climate_week','tot_climate_action_week', 'active_month', 'duration_month', 'active_year',
 'tot_natural_disaster_week','tot_news_week','duration','activation_week','duration_year',
 'tot_climate_month','tot_climate_action_month','tot_natural_disaster_month','tot_news_month','tot_news_year',
 'tot_climate_year', 'tot_climate_action_year', 'tot_natural_disaster_year'
 ], inplace = True, axis = 1)
    return df_sample

def get_lr_df(df_sample):
    df_lr = df_sample.copy()
    subreddits_features = [u+k for u in ft.subreddits for k in [p for p in ["_week", "_month", "_year"]]]
    r_subreddits_features = ["r" + u for u in subreddits_features]
    df_lr = df_lr.rename(columns = {u: "r" + u for u in subreddits_features})
    non_subreddit_features = [u for u in df_sample.columns if u not in subreddits_features]
    
    df_lr.fillna(df_lr.median(numeric_only = True), inplace = True)
    df_lr.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_lr.dropna(inplace=True)
    df_lr[[u for u in df_lr.columns if "norm" in u]] = df_lr[[u for u in df_lr.columns if "norm" in u]] * 100
    
    week_features = [u for u in non_subreddit_features if (u[-5:] == "_week")&("activation" not in u)&("duration" not in u)]
    year_features = [u[:-5] + "_year" for u in week_features]
    week_to_year_features = [u[:-5] + "_week_to_year" for u in week_features]
    df_week_to_year = ((df_lr[week_features]
                        .rename(columns = {u0: u1 for u0, u1 in zip(week_features, week_to_year_features)}).add(0.001))
                       .div(df_lr[[u[:-5] + "_year" for u in week_features]]
                            .rename(columns = {u0: u1 for u0, u1 in zip(year_features, week_to_year_features)}).add(0.01)))
    
    week_subreddit_features = [u for u in df_lr.columns if (u[-5:] == "_week")&(u in r_subreddits_features)]
    year_subreddit_features = [u[:-5] + "_year" for u in week_subreddit_features]
    week_not_year_subreddit_features = [u[:-5] + "_week_not_year" for u in week_subreddit_features]
    
    df_subreddits_week_not_year = ((df_lr[week_subreddit_features]
                                    .rename(columns = {u0: u1 for u0, u1 in zip(week_subreddit_features, week_not_year_subreddit_features)})
                                    .astype(float)
                                    - df_lr[[u[:-5] + "_year" for u in week_subreddit_features]]
                                    .rename(columns = {u0: u1 for u0, u1 in zip(year_subreddit_features, week_not_year_subreddit_features)})
                                    .astype(float)) > 0)
    
    df_lr = pd.concat([df_lr, df_week_to_year, df_subreddits_week_not_year], axis = 1)
    
    # df_lr = pd.concat([df_lr, df_week_to_year], axis = 1)
    # # df_lr = df_lr.loc[:, non_subreddit_features + subreddits_features[:60]]
    # df_lr = df_lr.loc[:, df_lr.var() > 0]
    
    # events = df_lr['activation_week_week'].astype(bool)
    # for (i,var_s1, var_s2) in zip(range(len(r_subreddits_features)),
    #                               (df_lr.loc[events,r_subreddits_features].var() > 0.01), 
    #                               (df_lr.loc[~events,r_subreddits_features].var() > 0.01)):
    #     if var_s1&var_s2:
    #         pass
    #     else:
    #         break
    # print(i - (i%3))
    # df_lr = df_lr.loc[:, non_subreddit_features + week_to_year_features + r_subreddits_features[:(i - (i%3))] + week_not_year_subreddit_features[:(i - (i%3))]]
    # df_lr = df_lr.loc[:, non_subreddit_features + week_to_year_features + subreddits_features[:(i - (i%3))]]
    # df_lr = df_lr.loc[:, non_subreddit_features + subreddits_features[:(i - (i%3))]]
    
    # if sum(df_lr.var() < 0.01) > 0:
    #     print(df_lr.columns[list(df_lr.var() < 0.01)])
    #     df_lr = df_lr.loc[:, df_lr.var() > 0.01]
    
    # print("Original shape", df_lr.shape)
    # print("Filtered shape", df_lr.shape)
    return df_lr

def get_bootstrap_df(df_sample, random_state = None):
    return df_sample.sample(len(df_sample), replace = True, random_state = random_state)


def exp_df_from_bootstrap(bootstrap_df):
    return bootstrap_df.apply(lambda x: (x.mean(), x.quantile(0.025), x.quantile(0.975))).T.rename(columns = {0: "coef", 1: "CI_lower", 2: "CI_upper"})
    


def logistic_regression_activation(df_lr, random_state = 1):
    X_train, X_test, y_train, y_test = train_test_split(df_lr.drop(["activation_week_week", "duration_week"], axis = 1),
                                                        df_lr["activation_week_week"], test_size=0.25, random_state=random_state)
    
    df_train = pd.concat([X_train, y_train], axis = 1).astype(float)
    model = smf.logit("activation_week_week" + " ~ " + " + ".join(X_train.columns), 
                      data = df_train)
    log_reg = model.fit(disp = False)
    return log_reg, X_train, X_test, y_train, y_test

def log_regression_analysis(log_reg, X_test, y_test):
    summary_logreg = log_reg.summary2().tables[1].rename(columns = {"Coef.": "coef", "P>|z|": "p"})[["coef", "p"]]
    y_pred = log_reg.predict(X_test.astype(float)).round()
    confusion_matrix = pd.DataFrame([np.array(y_test).astype(int), np.array(y_pred).astype(int)], index = ["y_test", "y_pred"]).T.reset_index().groupby(["y_test", "y_pred"]).count()["index"].unstack().loc[[1.,0.], [1.,0.]]
    positive_predictors, negative_predictors = summary_logreg.query("(p < 0.05)&(coef > 0)").index, summary_logreg.query("(p < 0.05)&(coef < 0)").index

    return summary_logreg, confusion_matrix, positive_predictors, negative_predictors

def confusion_matrix_analysis(cm):
    tot = cm.sum().sum()
    accuracy = (cm.loc[1,1] + cm.loc[0,0]) / tot
    recall = cm.loc[1,1] / cm.sum()[1]
    precision = cm.loc[1,1] / cm.sum(axis = 1)[1]
    random_baseline = (cm.sum() / cm.sum().sum()).max()
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "random_baseline": random_baseline}


def complete_experiment_log_reg(month_start, month_end, subreddit_class, save, df = None,
                                positive_size = None, negative_size = 1000, random_state = None, bootstrap = False):
    if df is None:
        path = cu.data_path + f"features_authors/{subreddit_class}_{month_start}_{month_end}.csv.gz"
        if os.path.exists(path):
            df = pd.read_csv(path, compression = "gzip", index_col = 0)
        else:
            df = get_features_authors(month_start, month_end, subreddit_class, save)
    df_sample = get_sample_df(df, random_state = random_state, positive_size = positive_size, negative_size = negative_size)
    if bootstrap:
        df_sample = get_bootstrap_df(df_sample, random_state = random_state)

    df_lr = get_lr_df(df_sample)
    log_reg, X_train, X_test, y_train, y_test = logistic_regression_activation(df_lr, random_state = random_state)
    summary_logreg, confusion_matrix, positive_predictors, negative_predictors = log_regression_analysis(log_reg, X_test, y_test)
    return {
        "summary_logreg": summary_logreg, 
        "confusion_matrix": confusion_matrix, 
        "positive_predictors": positive_predictors, 
        "negative_predictors": negative_predictors}



if __name__ == "__main__":
    subreddit_class = sys.argv[1]
    for k in range(22, 69):
        month_start, month_end = ft.all_months[k], ft.all_months[k+1]
        print(month_start)
        df = get_features_authors(month_start, month_end, subreddit_class, save = True)
















