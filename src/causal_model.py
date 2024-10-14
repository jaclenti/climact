import sys
sys.path += ["../src"]
import climact_shared.src.utils as cu
# import features_cox_week as ft
# import logistic_regression_activation as L
import pandas as pd
from glob import glob
import numpy as np
# import cox_logreg_experiments as cle
# import create_features_df
from sklearn.preprocessing import StandardScaler, Normalizer
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from matplotlib.pyplot import subplots as sbp

import jax, jaxlib
import jax.numpy as jnp
import jax.random as random
from jax.random import PRNGKey
from jax import lax

# from scipy.special import expit as np_sigmoid
import numpyro
from time import time
# from jax.scipy.special import expit as sigmoid
from numpyro.infer import SVI, Trace_ELBO, TraceGraph_ELBO, TraceMeanField_ELBO, MCMC, NUTS, Predictive, init_to_value
from numpyro.infer.autoguide import AutoNormal, AutoMultivariateNormal, AutoGuideList, AutoLowRankMultivariateNormal
from numpyro import distributions
from numpyro.optim import Adam
import dill
from collections import namedtuple

path_to_exp = "/data/shared/xxx/climact/experiments/"

features_interaction_short = [u for u in cu.features_join["interaction_short"] if "comment" in u]
scores = pd.read_csv("../data/scores.big.csv", index_col = 0)[["age", "gender", "partisan", "affluence"]]
subreddit_list = list(pd.read_csv(path_to_exp + "subreddit_list.csv")["0"])
SVIRunResult = namedtuple("SVIRunResult", ["params", "state", "losses"])

params_list = ['sociodemo_users',
               'beta_sociodemo_to_opinions',
               'beta_activity_to_opinions',
               'beta_news_to_opinions',
               'beta_intercept_to_opinions',
               
               'opinions_noise',
               
               'beta_sociodemo_to_sub_LT',
               'beta_activity_to_sub_LT',
               'beta_popularity_to_sub_LT',
               'beta_intercept_to_sub_LT',
               
               'beta_activity_to_activity_ST',
               'beta_intercept_to_activity_ST', 
               
               'beta_retention_to_sub_ST',
               'beta_sociodemo_to_sub_ST',
               'beta_activity_to_sub_ST',
               'beta_popularity_to_sub_ST',
               'beta_intercept_to_sub_ST',
               
               'beta_opinion_to_interaction',
               'beta_activity_to_interaction',
               'beta_sociodemo_to_interaction',
               'beta_intercept_to_interaction',
               
               'beta_interaction_to_activation',
               'beta_opinion_to_activation',
               'beta_activity_to_activation',
               'beta_news_LT_to_activation',
               'beta_news_ST_to_activation',
               'beta_intercept_to_activation',]

def load_dataframe(subreddit_class):
    # lr = L.LogisticRegression_activation(subreddit_class) 
    df = pd.read_csv(cu.data_path + f"authors_featurs_df/{subreddit_class}.csv.gz", 
                        compression = "gzip", index_col = [0,1])
    df[cu.features_join["interaction_short"]] = df[cu.features_join["interaction_short"]] > 0

    return df
    
def is_significant(v1, v2, dist, confidence_interval = 0.9):
    samples = (dist(v1,v2).sample(PRNGKey(1), sample_shape = (10000,)) > 0).mean(axis = 0)
    significant = (samples < ((1 - confidence_interval) / 2)) | (samples > (1 - (1 - confidence_interval) / 2))
    return significant

def get_data(df, scores, remove_non_active = False):
    df[[u for u in cu.features_join["subreddit_long"] if u not in df.columns]] = False
    df[[u for u in cu.features_join["subreddit_short"] if u not in df.columns]] = False
    subreddit_list = sorted([u for u in scores.index if (f"r{u}_long" in df.columns )&(f"r{u}_short" in df.columns)])
    features_subreddit_long = [f"r{u}_long" for u in subreddit_list]
    features_subreddit_short = [f"r{u}_short" for u in subreddit_list]
    if remove_non_active:
        df = df.loc[(df[features_subreddit_long].sum(axis = 1) > 0)&(df[features_subreddit_short].sum(axis = 1) > 0)]
    
    
    scaler = StandardScaler()
    act_LT, act_ST = (df[cu.features_join["control_long"]]), (df[cu.features_join["control_short"]])
    log_act_LT, log_act_ST = np.log(act_LT + 1.), np.log(act_ST + 1.)
    scaler.fit(act_LT)
    activity_LT = jnp.array(scaler.transform(act_LT))
    scaler.fit(log_act_LT)
    log_activity_LT = jnp.array(scaler.transform(log_act_LT))
    scaler.fit(act_ST)
    activity_ST = scaler.transform(act_ST)
    scaler.fit(log_act_ST)
    log_activity_ST = scaler.transform(log_act_ST)
    
    scaler.fit(df[cu.features_join["norm_news_long"]])
    news_LT = jnp.array(scaler.transform(df[cu.features_join["norm_news_long"]]))
    scaler.fit(df[cu.features_join["norm_news_short"]])
    news_ST = jnp.array(scaler.transform(df[cu.features_join["norm_news_short"]]))
    
    sociodemo_subreddit = jnp.array(scores.loc[subreddit_list])
    log_pop = np.array(np.log(df[features_subreddit_long].sum() + .1))[:,None]
    scaler.fit(log_pop)
    log_popularity_subreddits = jnp.array(scaler.transform(log_pop)).T
    pop = np.array(df[features_subreddit_long].sum())[:,None]
    scaler.fit(pop)
    popularity_subreddits = jnp.array(scaler.transform(pop)).T
    
    subreddit_LT = jnp.array(df[features_subreddit_long])
    subreddit_ST = jnp.array(df[features_subreddit_short])
    interaction_ST = jnp.array(df['n_different_comments_with_active_link_id_id_short']|df['n_different_comments_with_active_id_link_id_short'])
    
    activation = jnp.array(df["activation"])[:,None]
    
    return ((subreddit_LT, activity_LT, log_activity_LT, news_LT, subreddit_ST, 
            activity_ST, log_activity_ST, news_ST, interaction_ST, activation, 
            sociodemo_subreddit, popularity_subreddits, log_popularity_subreddits, subreddit_list), 
            (subreddit_LT, activity_ST, subreddit_ST, interaction_ST, activation))
    
def ex_model(data, data_obs):
    # subreddit_LT, activity_LT, log_activity_LT, news_LT, subreddit_ST, activity_ST, log_activity_ST, news_ST, interaction_ST, activation, sociodemo_subreddit, popularity_subreddits, log_popularity_subreddits, subreddit_list = data
    # subreddit_LT_obs, activity_ST_obs, subreddit_ST_obs, interaction_ST_obs, activation_obs = data_obs
    
    # act_dim = 10
    # news_dim = 3
    # socio_dim = 4

    # full_activity_LT = jnp.concat([activity_LT, log_activity_LT], axis = 1)
    # full_activity_ST = jnp.concat([activity_ST, log_activity_ST], axis = 1)
    # full_popularity_subreddits = jnp.concat([popularity_subreddits, log_popularity_subreddits], axis = 0).T

    # n_authors, n_subreddits = subreddit_LT.shape
    
    # sociodemo_users = numpyro.sample("sociodemo_users",  distributions.Normal(jnp.zeros((n_authors, socio_dim))))
    # sociodemo_users = (sociodemo_users - jnp.mean(sociodemo_users, axis = 0)) / jnp.std(sociodemo_users, axis = 0)

    # beta_sociodemo_to_opinions = numpyro.sample("beta_sociodemo_to_opinions", distributions.MultivariateNormal(jnp.zeros(socio_dim), covariance_matrix = jnp.eye(socio_dim,socio_dim)))
    # beta_activity_to_opinions = numpyro.sample("beta_activity_to_opinions", distributions.MultivariateNormal(jnp.zeros((act_dim)), jnp.eye(act_dim,act_dim)))
    # beta_news_to_opinions = numpyro.sample("beta_news_to_opinions", distributions.MultivariateNormal(jnp.zeros((news_dim)), jnp.eye(news_dim,news_dim)))
    
    # opinions_median = (sociodemo_users @ beta_sociodemo_to_opinions + full_activity_LT @ beta_activity_to_opinions + news_LT @ beta_news_to_opinions)[:,None]
    # opinions_noise = numpyro.sample("opinions_noise", distributions.Normal(jnp.zeros((n_authors)), 0.1 * jnp.ones((n_authors))))[:,None]
    # opinions = opinions_median + opinions_noise
    
    # beta_sociodemo_to_sub_LT = numpyro.sample("beta_sociodemo_to_sub_LT", distributions.MultivariateNormal(jnp.zeros((socio_dim)), jnp.eye(socio_dim,socio_dim)))
    # beta_activity_to_sub_LT = numpyro.sample("beta_activity_to_sub_LT", distributions.MultivariateNormal(jnp.zeros((act_dim)), jnp.eye(act_dim,act_dim)))
    # beta_popularity_to_sub_LT = numpyro.sample("beta_popularity_to_sub_LT", distributions.Normal(jnp.zeros((2))))
    # beta_intercept_to_sub_LT = numpyro.sample("beta_intercept_to_sub_LT", distributions.Normal(jnp.array([0.])))

    # logits_subreddit_LT = beta_sociodemo_to_sub_LT * sociodemo_users @ sociodemo_subreddit.T + full_popularity_subreddits @ beta_popularity_to_sub_LT + (full_activity_LT @ beta_activity_to_sub_LT)[:,None] +  beta_intercept_to_sub_LT
        
    # beta_activity_to_activity_ST = numpyro.sample("beta_activity_to_activity_ST", distributions.Normal(jnp.zeros((5)))) 
    # beta_intercept_to_activity_ST = numpyro.sample("beta_intercept_to_activity_ST", distributions.Normal(jnp.zeros((5))))
    
    # median_activity_ST = beta_activity_to_activity_ST * activity_LT + beta_intercept_to_activity_ST
    
    # beta_retention_to_sub_ST = numpyro.sample("beta_retention_to_sub_ST", distributions.Normal(jnp.zeros(1)))
    # beta_sociodemo_to_sub_ST = numpyro.sample("beta_sociodemo_to_sub_ST", distributions.MultivariateNormal(jnp.zeros((socio_dim)), jnp.eye(socio_dim,socio_dim)))
    # beta_activity_to_sub_ST = numpyro.sample("beta_activity_to_sub_ST", distributions.MultivariateNormal(jnp.zeros((act_dim)), jnp.eye(act_dim,act_dim)))
    # beta_popularity_to_sub_ST = numpyro.sample("beta_popularity_to_sub_ST", distributions.Normal(jnp.zeros((2))))
    # beta_intercept_to_sub_ST = numpyro.sample("beta_intercept_to_sub_ST", distributions.Normal(jnp.array([0.])))
    
    # logits_subreddit_ST = subreddit_LT * beta_retention_to_sub_ST + beta_sociodemo_to_sub_ST * opinions @ sociodemo_subreddit.T  + (full_activity_ST @ beta_activity_to_sub_ST)[:,None] + (full_popularity_subreddits @ beta_popularity_to_sub_ST)[None,:] + beta_intercept_to_sub_ST
    
    # beta_activity_to_interaction = numpyro.sample("beta_activity_to_interaction", distributions.MultivariateNormal(jnp.zeros((act_dim)), jnp.eye(act_dim,act_dim)))
    # beta_sociodemo_to_interaction = numpyro.sample("beta_sociodemo_to_interaction", distributions.Normal(jnp.zeros(socio_dim)))
    # beta_intercept_to_interaction = numpyro.sample("beta_intercept_to_interaction", distributions.Normal(jnp.array([0.])))
    
    # logits_interaction_ST = (full_activity_ST @ beta_activity_to_interaction)[:,None] + (subreddit_ST @ sociodemo_subreddit @ beta_sociodemo_to_interaction[:,None]) + beta_intercept_to_interaction
    # # beta_opinion_to_interaction = numpyro.sample("beta_opinion_to_interaction", distributions.Normal(jnp.array([0.])))
    # # logits_interaction_ST = (beta_opinion_to_interaction * opinions) + (activity_ST @ beta_activity_to_interaction)[:,None] + (log_activity_ST @ beta_log_activity_to_interaction)[:,None] + (subreddit_ST @ sociodemo_subreddit @ beta_sociodemo_to_interaction[:,None]) + beta_intercept_to_interaction
    
    # beta_interaction_to_activation = numpyro.sample("beta_interaction_to_activation", distributions.Normal(jnp.zeros((1))))
    # beta_opinion_to_activation = numpyro.sample("beta_opinion_to_activation", distributions.Normal(jnp.array([0.])))
    # beta_activity_to_activation = numpyro.sample("beta_activity_to_activation", distributions.MultivariateNormal(jnp.zeros((act_dim)), jnp.eye(act_dim,act_dim)))
    # beta_news_LT_to_activation = numpyro.sample("beta_news_LT_to_activation", distributions.MultivariateNormal(jnp.zeros((news_dim)), jnp.eye(news_dim,news_dim)))
    # beta_news_ST_to_activation = numpyro.sample("beta_news_ST_to_activation", distributions.MultivariateNormal(jnp.zeros((news_dim)), jnp.eye(news_dim,news_dim)))
    # beta_intercept_to_activation = numpyro.sample("beta_intercept_to_activation", distributions.Normal(jnp.array((0.))))
    # logits_activation = (interaction_ST * beta_interaction_to_activation)[:,None] + opinions * beta_opinion_to_activation + (news_ST @ beta_news_ST_to_activation)[:,None] + (full_activity_ST @ beta_activity_to_activation)[:,None] + (news_LT @ beta_news_LT_to_activation)[:,None] + beta_intercept_to_activation
    
    
    # numpyro.sample("subreddit_LT",
    #                distributions.BernoulliLogits(logits_subreddit_LT).to_event(1),
    #                obs = subreddit_LT_obs)
    # numpyro.sample("activity_ST", 
    #                distributions.Normal(median_activity_ST).to_event(1),
    #                obs = activity_ST_obs)
    # numpyro.sample("subreddit_ST", 
    #                distributions.BernoulliLogits(logits_subreddit_ST).to_event(1),
    #                obs = subreddit_ST_obs)
    # numpyro.sample("interaction_ST", 
    #                distributions.BernoulliLogits(logits_interaction_ST).to_event(1),
    #                obs = interaction_ST_obs)
    # numpyro.sample("activation",
    #                distributions.BernoulliLogits(logits_activation).to_event(1),
    #                obs = activation_obs)
    subreddit_LT, activity_LT, log_activity_LT, news_LT, subreddit_ST, activity_ST, log_activity_ST, news_ST, interaction_ST, activation, sociodemo_subreddit, popularity_subreddits, log_popularity_subreddits, subreddit_list = data

def model(data, data_obs, var_opinions = 0.1):
    subreddit_LT, activity_LT, log_activity_LT, news_LT, subreddit_ST, activity_ST, log_activity_ST, news_ST, interaction_ST, activation, sociodemo_subreddit, popularity_subreddits, log_popularity_subreddits, subreddit_list = data
    subreddit_LT_obs, activity_ST_obs, subreddit_ST_obs, interaction_ST_obs, activation_obs = data_obs
    if interaction_ST_obs is not None:
       interaction_ST_obs = interaction_ST_obs[:,None]
    
    act_dim = 1
    news_dim = 3
    socio_dim = 4
    pop_dim = 1

    full_activity_LT = jnp.concat([activity_LT, log_activity_LT], axis = 1)
    full_activity_ST = jnp.concat([activity_ST, log_activity_ST], axis = 1)
    full_popularity_subreddits = jnp.concat([popularity_subreddits, log_popularity_subreddits], axis = 0).T

    full_activity_LT = log_activity_LT[:,3][:,None]
    full_activity_ST = log_activity_ST[:,3][:,None]
    full_popularity_subreddits = full_popularity_subreddits[:,1][:,None]

    n_authors, n_subreddits = subreddit_LT.shape
    
    sociodemo_users = numpyro.sample("sociodemo_users",  distributions.Normal(jnp.zeros((n_authors, socio_dim))))
    sociodemo_users = (sociodemo_users - jnp.mean(sociodemo_users, axis = 0)) / jnp.std(sociodemo_users, axis = 0)

    beta_sociodemo_to_opinions = numpyro.sample("beta_sociodemo_to_opinions", distributions.MultivariateNormal(jnp.zeros(socio_dim), covariance_matrix = jnp.eye(socio_dim,socio_dim)))
    beta_activity_to_opinions = numpyro.sample("beta_activity_to_opinions", distributions.MultivariateNormal(jnp.zeros((act_dim)), jnp.eye(act_dim,act_dim)))
    beta_news_to_opinions = numpyro.sample("beta_news_to_opinions", distributions.MultivariateNormal(jnp.zeros((news_dim)), jnp.eye(news_dim,news_dim)))
    beta_intercept_to_opinions = numpyro.sample("beta_intercept_to_opinions", distributions.Normal(jnp.array([0.])))
    
    opinions_median = (sociodemo_users @ beta_sociodemo_to_opinions + full_activity_LT @ beta_activity_to_opinions + news_LT @ beta_news_to_opinions)[:,None]
    opinions_noise = numpyro.sample("opinions_noise", distributions.Normal(jnp.zeros((n_authors)), np.sqrt(var_opinions) * jnp.ones((n_authors))))[:,None]
    opinions = opinions_median + opinions_noise
    opinions = (opinions - opinions.mean()) / opinions.std()
    
    beta_sociodemo_to_sub_LT = numpyro.sample("beta_sociodemo_to_sub_LT", distributions.MultivariateNormal(jnp.zeros((socio_dim)), jnp.eye(socio_dim,socio_dim)))
    beta_activity_to_sub_LT = numpyro.sample("beta_activity_to_sub_LT", distributions.MultivariateNormal(jnp.zeros((act_dim)), jnp.eye(act_dim,act_dim)))
    beta_popularity_to_sub_LT = numpyro.sample("beta_popularity_to_sub_LT", distributions.Normal(jnp.zeros((pop_dim))))
    beta_intercept_to_sub_LT = numpyro.sample("beta_intercept_to_sub_LT", distributions.Normal(jnp.array([0.])))

    logits_subreddit_LT = beta_sociodemo_to_sub_LT * sociodemo_users @ sociodemo_subreddit.T + full_popularity_subreddits @ beta_popularity_to_sub_LT + (full_activity_LT @ beta_activity_to_sub_LT)[:,None] +  beta_intercept_to_sub_LT
        
    beta_activity_to_activity_ST = numpyro.sample("beta_activity_to_activity_ST", distributions.Normal(jnp.zeros((act_dim)))) 
    beta_intercept_to_activity_ST = numpyro.sample("beta_intercept_to_activity_ST", distributions.Normal(jnp.zeros((1))))
    
    median_activity_ST = beta_activity_to_activity_ST * activity_LT + beta_intercept_to_activity_ST
    
    beta_retention_to_sub_ST = numpyro.sample("beta_retention_to_sub_ST", distributions.Normal(jnp.zeros(1)))
    beta_sociodemo_to_sub_ST = numpyro.sample("beta_sociodemo_to_sub_ST", distributions.MultivariateNormal(jnp.zeros((socio_dim)), jnp.eye(socio_dim,socio_dim)))
    beta_activity_to_sub_ST = numpyro.sample("beta_activity_to_sub_ST", distributions.MultivariateNormal(jnp.zeros((act_dim)), jnp.eye(act_dim,act_dim)))
    beta_popularity_to_sub_ST = numpyro.sample("beta_popularity_to_sub_ST", distributions.Normal(jnp.zeros((pop_dim))))
    beta_intercept_to_sub_ST = numpyro.sample("beta_intercept_to_sub_ST", distributions.Normal(jnp.array([0.])))
    
    logits_subreddit_ST = subreddit_LT * beta_retention_to_sub_ST + beta_sociodemo_to_sub_ST * opinions @ sociodemo_subreddit.T  + (full_activity_ST @ beta_activity_to_sub_ST)[:,None] + (full_popularity_subreddits @ beta_popularity_to_sub_ST)[None,:] + beta_intercept_to_sub_ST
    
    beta_opinion_to_interaction = numpyro.sample("beta_opinion_to_interaction", distributions.Normal(jnp.array([0.])))
    beta_activity_to_interaction = numpyro.sample("beta_activity_to_interaction", distributions.MultivariateNormal(jnp.zeros((act_dim)), jnp.eye(act_dim,act_dim)))
    # beta_sociodemo_to_interaction = numpyro.sample("beta_sociodemo_to_interaction", distributions.MultivariateNormal(jnp.zeros(socio_dim), jnp.eye(socio_dim,socio_dim)))
    beta_sociodemo_to_interaction = numpyro.sample("beta_sociodemo_to_interaction", distributions.Normal(jnp.zeros(socio_dim)))
    beta_intercept_to_interaction = numpyro.sample("beta_intercept_to_interaction", distributions.Normal(jnp.array([0.])))
    
    logits_interaction_ST = (full_activity_ST @ beta_activity_to_interaction)[:,None] + (subreddit_ST @ sociodemo_subreddit @ beta_sociodemo_to_interaction[:,None]) + beta_intercept_to_interaction
         
    beta_interaction_to_activation = numpyro.sample("beta_interaction_to_activation", distributions.Normal(jnp.zeros((1))))
    beta_opinion_to_activation = numpyro.sample("beta_opinion_to_activation", distributions.Normal(jnp.array([0.])))
    beta_activity_to_activation = numpyro.sample("beta_activity_to_activation", distributions.MultivariateNormal(jnp.zeros((act_dim)), jnp.eye(act_dim,act_dim)))
    beta_news_LT_to_activation = numpyro.sample("beta_news_LT_to_activation", distributions.MultivariateNormal(jnp.zeros((news_dim)), jnp.eye(news_dim,news_dim)))
    beta_news_ST_to_activation = numpyro.sample("beta_news_ST_to_activation", distributions.MultivariateNormal(jnp.zeros((news_dim)), jnp.eye(news_dim,news_dim)))
    beta_intercept_to_activation = numpyro.sample("beta_intercept_to_activation", distributions.Normal(jnp.array((0.))))
    logits_activation = (interaction_ST * beta_interaction_to_activation)[:,None] + opinions * beta_opinion_to_activation + (news_ST @ beta_news_ST_to_activation)[:,None] + (full_activity_ST @ beta_activity_to_activation)[:,None] + (news_LT @ beta_news_LT_to_activation)[:,None] + beta_intercept_to_activation
    
    
    numpyro.sample("subreddit_LT",
                   distributions.BernoulliLogits(logits_subreddit_LT).to_event(1),
                   obs = subreddit_LT_obs)
    numpyro.sample("activity_ST", 
                   distributions.Normal(median_activity_ST).to_event(1),
                   obs = activity_ST_obs)
    numpyro.sample("subreddit_ST", 
                   distributions.BernoulliLogits(logits_subreddit_ST).to_event(1),
                   obs = subreddit_ST_obs)
    numpyro.sample("interaction_ST", 
                   distributions.BernoulliLogits(logits_interaction_ST).to_event(1),
                   obs = interaction_ST_obs)
    numpyro.sample("activation",
                   distributions.BernoulliLogits(logits_activation).to_event(1),
                   obs = activation_obs)
# data, data_obs = get_data(df, scores)
# subreddit_LT, activity_LT, log_activity_LT, news_LT, subreddit_ST, activity_ST, log_activity_ST, news_ST, interaction_ST, activation, sociodemo_subreddit, popularity_subreddits, log_popularity_subreddits, subreddit_list = data
# full_activity_LT = jnp.concat([activity_LT, log_activity_LT], axis = 1)
# full_activity_ST = jnp.concat([activity_ST, log_activity_ST], axis = 1)
# full_popularity_subreddits = jnp.concat([popularity_subreddits, log_popularity_subreddits], axis = 0).T

# n_authors, n_subreddits = subreddit_LT.shape

def get_init_params(n_authors):
    socio_dim = 4
    act_dim = 1
    news_dim = 3
    pop_dim = 1
    
    init_params_normal = {"beta_intercept_to_opinions": jnp.zeros((1)),
                        "sociodemo_users": jnp.zeros((n_authors, socio_dim)),
                        "opinions_noise": jnp.zeros((n_authors)),
                        "beta_popularity_to_sub_LT": jnp.ones(pop_dim),
                        "beta_intercept_to_sub_LT": jnp.array([-2.]),
                        "beta_activity_to_activity_ST": jnp.ones(act_dim),
                        "beta_intercept_to_activity_ST": jnp.zeros(1),
                        "beta_retention_to_sub_ST": jnp.array([1.]),
                        "beta_popularity_to_sub_ST": jnp.ones(pop_dim),
                        "beta_intercept_to_sub_ST": jnp.array([-2.]),
                        "beta_opinion_to_interaction": jnp.array([1.]),
                        "beta_sociodemo_to_interaction": jnp.zeros((socio_dim)),
                        "beta_intercept_to_interaction": jnp.array([0.]),
                        "beta_opinion_to_activation": jnp.array([1.]),
                        "beta_intercept_to_activation": jnp.zeros((1)),
                        "beta_interaction_to_activation": jnp.array([1.]),}
    init_params_multivariatenormal = {"beta_sociodemo_to_sub_LT": jnp.repeat(10., socio_dim),
                                      "beta_activity_to_sub_LT": jnp.zeros((act_dim)),
                                      "beta_sociodemo_to_sub_ST": jnp.repeat(0., socio_dim),
                                      "beta_activity_to_sub_ST": jnp.zeros((act_dim)),
                                      "beta_sociodemo_to_opinions": jnp.zeros((socio_dim)),
                                      "beta_activity_to_opinions": jnp.zeros((act_dim)),
                                      "beta_news_to_opinions": jnp.zeros((news_dim)),
                                      "beta_activity_to_interaction": jnp.zeros((act_dim)),
                                    #   "beta_sociodemo_to_interaction": jnp.zeros((socio_dim)),
                                      "beta_activity_to_activation": jnp.zeros((act_dim)),
                                      "beta_news_LT_to_activation": jnp.zeros((news_dim)),
                                      "beta_news_ST_to_activation": jnp.zeros((news_dim))}
    
    return init_params_normal, init_params_multivariatenormal

def initialize_params(init_params_normal, init_params_multivariatenormal):
    init_params = {}
    
    for k in init_params_normal:
        init_params[k + "_auto_scale"] = 0.1 * jnp.ones(init_params_normal[k].shape)
        init_params[k + "_auto_loc"] = init_params_normal[k]
    
    for k in init_params_multivariatenormal:
        init_params[k + "_scale_tril"] = 0.1 * jnp.eye(len(init_params_multivariatenormal[k]), len(init_params_multivariatenormal[k]))
        init_params[k + "_loc"] = init_params_multivariatenormal[k]
    return init_params

def get_guide(model, model_settings, data, data_obs, init_params_normal, init_params_multivariatenormal, var_opinions = 0.1, multivariate = True, init_scale = 0.1, init_scale_opinions = 0.1):
    if multivariate:
        guide = AutoGuideList(model)
        for param in params_list:
            scale = init_scale_opinions if param == "opinions_noise" else init_scale
            if param in init_params_normal.keys():
                guide.append(AutoNormal(numpyro.handlers.block(numpyro.handlers.seed(model, PRNGKey(12)), 
                                                               expose = [param]),
                                                               init_loc_fn = init_to_value(values = {param: init_params_normal[param]}),
                                                               init_scale = scale))
            if param in init_params_multivariatenormal.keys():
                guide.append(AutoMultivariateNormal(numpyro.handlers.block(numpyro.handlers.seed(model, PRNGKey(12)), expose = [param]), 
                                                    prefix = param,
                                                    init_loc_fn = init_to_value(values = {param: init_params_multivariatenormal[param]})))
    else:
        guide = AutoNormal(model)
    optimizer = Adam(step_size = model_settings["lr"])
    svi = SVI(model, guide, optimizer, loss = TraceMeanField_ELBO())
    svi.init(PRNGKey(100), data, data_obs, var_opinions)
    
    return guide, svi

def get_prior_samples(guide, data, init_params, n_samples, key):
    prior_samples = guide.sample_posterior(PRNGKey(key), params = init_params, sample_shape = (n_samples,))
    prior_predictive = Predictive(model, prior_samples, num_samples = n_samples)
    prior_predictions = prior_predictive(PRNGKey(key), data, (None,None,None,None,None))
    return prior_samples, prior_predictions

def run_early_stop(self, rng_key, num_steps, intermediate_steps = 10,
          *args, init_state = None, init_params = None, delta_params = 10,
          delta_loss = 1, max_no_update = 5, return_losses = True, **kwargs,):

    def body_fn(svi_state, _):
        svi_state, loss = self.update(
            svi_state,
            *args,
            forward_mode_differentiation=False,
            **kwargs,
            )
        return svi_state, loss

    if init_state is None:
        svi_state = self.init(rng_key, *args, init_params=init_params, **kwargs)
    else:
        svi_state = init_state
    svi_res1 = None
    loss1 = None
    all_losses = jnp.array([])
    no_update_steps = 0
    for _ in range(int(num_steps / intermediate_steps)):
        svi_res0 = svi_res1
        loss0 = loss1
        svi_state, losses = lax.scan(body_fn, svi_state, None, length = intermediate_steps)
        loss1 = losses[-1]
        all_losses = jnp.append(all_losses, losses)
        svi_res1 = SVIRunResult(self.get_params(svi_state), svi_state, losses)
        if _ > 0:
            dist_params = sum([((svi_res1.params[v] - svi_res0.params[v])**2).sum() for v in svi_res1.params])
            dist_loss = (loss0 - loss1)
            
            if  (dist_params < delta_params)|((dist_loss < delta_loss)&(dist_loss > 0)):
                no_update_steps += 1
                if no_update_steps > max_no_update:
                    break
    if return_losses:
        print(len(all_losses))
        return svi_res1, all_losses
    else:
        return svi_res1

def get_accuracy(guide, svi_results, subreddit_class = None, data = None):
    if subreddit_class:
        df = load_dataframe(subreddit_class)
        data, data_obs = get_data(df, scores)
    samples = sample_posterior(guide, svi_results.params, data)
    accuracies = (samples["activation"] == (data[-5]+0.)).mean(axis = 1)
    return accuracies.mean().item(), accuracies.std().item()

def train(svi, guide, data, data_obs, model_settings, init_params_normal, init_params_multivariatenormal, var_opinions = 0.1, progress_bar = False, path = None, 
          print_loss = True, early_stop = True, key = 1000, early_stop_params = {"delta_loss": 100, "delta_params": 1.}):
    key = np.random.randint(low = 1, high = 1000)
    t0 = time()
    svi.run_early_stop = run_early_stop.__get__(svi)
    if early_stop:
        svi_results, losses = svi.run_early_stop(random.PRNGKey(key), model_settings["n_epochs"], data = data, data_obs = data_obs, 
                                                 var_opinions = var_opinions, delta_loss = early_stop_params["delta_loss"], delta_params = early_stop_params["delta_params"])
    else:                                             
        svi_results = svi.run(random.PRNGKey(key), model_settings["n_epochs"], data = data, data_obs = data_obs, var_opinions = var_opinions, progress_bar = progress_bar)
        losses = svi_results.losses
    t1 = time()
    model_settings["time"] = t1 - t0
    model_settings["num_epochs_early_stop"] = len(losses)
    model_settings["loss"] = losses[-1].item()
    accuracy_mean, accuracy_std = get_accuracy(guide, svi_results, data = data)
    model_settings["accuracy_mean"], model_settings["accuracy_std"] = accuracy_mean, accuracy_std
    betas = get_trained_beta(svi_results.params, data, data_obs, init_params_normal, init_params_multivariatenormal)
    if print_loss:
        print(losses[-1])
        fig, axes = sbp(ncols = 2, figsize = (4,2))
        axes[0].plot(losses)
        axes[1].plot(jnp.abs(jnp.diff(losses)))
        axes[1].set_yscale("log")
    if path:
        save_model(guide, svi_results, model_settings, path, betas)

    return guide, svi_results, betas, model_settings

def save_model(guide, svi_results, model_settings, path, betas):
    output_dict = {}
    
    output_dict["guide"] = guide
    output_dict["model_settings"] = model_settings
    output_dict["params"] = svi_results.params
    output_dict["losses"] = svi_results.losses
    output_dict["betas"] = cu.jnp_series_to_pd(betas)

    with open(path, "wb") as handle:
        dill.dump(output_dict, handle)

def load_model(path):
    with open(path, "rb") as f:
        input_dict = dill.load(f)
    
    return input_dict
    
def get_experiments_data(file):
    model_settings, betas,  = pd.read_pickle(file)["model_settings"], pd.read_pickle(file)["betas"]
    model_settings["loss"] = pd.read_pickle(file)["losses"][-1].item()
    setting_vars = ["subreddit_class", "lr", "var_opinions", "init_scale", "init_scale_opinions", "num_epochs_early_stop", "loss", "time"]
    settings_s = pd.Series([model_settings[k] for k in setting_vars], index = setting_vars)
    opinions_s = betas["beta_to_opinions"][[u for u in betas["beta_to_opinions"].index if (u in cu.sociodemo_classes) or ("norm" in u)]].rename(lambda x: x + "_to_opinion")
    activation_s = betas["beta_to_activation"][:8].rename(lambda x: x + "_to_activation")
    interaction_s = betas["beta_to_interaction"][cu.sociodemo_classes].rename(lambda x: x + "_to_interaction")
    return pd.concat([settings_s, opinions_s, activation_s, interaction_s])

def get_interesting_from_beta(betas):
    opinions_s = betas["beta_to_opinions"][[u for u in betas["beta_to_opinions"].index if (u in cu.sociodemo_classes) or ("norm" in u)]].rename(lambda x: x + "_to_opinion")
    activation_s = betas["beta_to_activation"][:8].rename(lambda x: x + "_to_activation")
    interaction_s = betas["beta_to_interaction"][cu.sociodemo_classes].rename(lambda x: x + "_to_interaction")
    return pd.concat([opinions_s, activation_s, interaction_s])

def param_from_dict_params(var_from, var_to, dict_params, significant_params):
    var_from_list_dict = [u for u in dict_params if ("beta_" + var_from in u ) & ("to_" + var_to in u) & ("_loc" in u)]
    var_from_list_sign = [u for u in significant_params if ("beta_" + var_from in u ) & ("to_" + var_to in u)]
    if len(var_from_list_dict) + len(var_from_list_sign) > 2:
        print(f"more than one {var_from} to {var_to}")
    prod = dict_params[var_from_list_dict[0]] * significant_params[var_from_list_sign[0]]
    if prod.shape == ():
        return [prod]
    else:
        return list(prod)

def get_trained_beta(params, data, data_obs, init_params_normal, init_params_multivariatenormal):
    subreddit_LT, activity_LT, log_activity_LT, news_LT, subreddit_ST, activity_ST, log_activity_ST, news_ST, interaction_ST, activation, sociodemo_subreddit, popularity_subreddits, log_popularity_subreddits, subreddit_list = data
    full_activity_LT = jnp.concat([activity_LT, log_activity_LT], axis = 1)[:,3][:,None]
    full_activity_ST = jnp.concat([activity_ST, log_activity_ST], axis = 1)
    full_popularity_subreddits = jnp.concat([popularity_subreddits, log_popularity_subreddits], axis = 0)[:,1][:,None]#.T

    dict_params = params
    
    significant_params = {p: is_significant(dict_params[p + "_auto_loc"], dict_params[p + "_auto_scale"], distributions.Normal) for p in init_params_normal}
    significant_params.update({p: is_significant(dict_params[p + "_loc"], dict_params[p + "_scale_tril"], distributions.MultivariateNormal) for p in init_params_multivariatenormal})
    
    sociodemo_users = pd.DataFrame(dict_params["sociodemo_users_auto_loc"], columns = scores.columns)
    sociodemo_users = (sociodemo_users - sociodemo_users.mean()) / sociodemo_users.std()
    opinions_median = pd.Series(dict_params["sociodemo_users_auto_loc"] @ dict_params["beta_sociodemo_to_opinions_loc"] + full_activity_LT @ dict_params["beta_activity_to_opinions_loc"] + news_LT @ dict_params["beta_news_to_opinions_loc"] + dict_params["beta_intercept_to_opinions_auto_loc"])
    opinions_users = opinions_median + dict_params["opinions_noise_auto_loc"]
    opinions_users = (opinions_users - opinions_users.mean()) / opinions_users.std()

    beta_to_opinions = (pd.Series(param_from_dict_params("sociodemo","opinions", dict_params, significant_params) 
                                  + param_from_dict_params("activity","opinions", dict_params, significant_params) 
                                  + param_from_dict_params("news","opinions", dict_params, significant_params)
                                  + param_from_dict_params("intercept","opinions", dict_params, significant_params), 
                                  index = list(scores.columns)
                                #   + cu.features_join["control_long"]
                                #   + ["log_" + u for u in cu.features_join["control_long"]]
                                  + ["n_comments_long"]
                                  + cu.features_join["norm_news_long"]
                                  + ["intercept"]))
    beta_to_activity_ST = (pd.Series(param_from_dict_params("activity","activity_ST", dict_params, significant_params)
                                     + param_from_dict_params("intercept","activity_ST", dict_params, significant_params),
                                     index = ["n_comments_long","intercept_n_comments_short"]))
                                    #  index = cu.features_join["control_long"] + ["intercept_" + u for u in cu.features_join["control_short"]]))

    beta_to_subreddit_LT = (pd.Series(param_from_dict_params("sociodemo","sub_LT",dict_params, significant_params)
                                      + param_from_dict_params("activity","sub_LT",dict_params, significant_params)
                                      + param_from_dict_params("popularity","sub_LT", dict_params, significant_params)
                                      + param_from_dict_params("intercept","sub_LT", dict_params, significant_params),
                                      index = list(scores.columns) + ["n_comments"] + ["log_popularity", "intercept"]))
                                    #   index = list(scores.columns) + cu.features_join["control_long"] + ["log_" + u for u in cu.features_join["control_long"]] + ["popularity", "log_popularity", "intercept"]))

    beta_to_subreddit_ST = (pd.Series(param_from_dict_params("sociodemo","sub_ST",dict_params, significant_params)
                                      + param_from_dict_params("retention","sub_ST", dict_params, significant_params)
                                      + param_from_dict_params("activity","sub_ST",dict_params, significant_params)
                                      + param_from_dict_params("popularity","sub_ST", dict_params, significant_params)
                                      + param_from_dict_params("intercept","sub_ST", dict_params, significant_params),
                                      index = list(scores.columns) + ["retention"] + ["n_comments_short"] + ["log_popularity", "intercept"]))
                                    #   index = list(scores.columns) + ["retention"] + cu.features_join["control_short"] + ["log_" + u for u in cu.features_join["control_short"]] + ["popularity", "log_popularity", "intercept"]))

    beta_to_interaction = (pd.Series(param_from_dict_params("opinion","interaction", dict_params, significant_params)
                                     +  param_from_dict_params("activity","interaction",dict_params, significant_params)
                                     + param_from_dict_params("sociodemo","interaction", dict_params, significant_params)
                                     + param_from_dict_params("intercept","interaction", dict_params, significant_params),
                                     index = ["opinion"] + ["n_comments_short"]  + list(scores.columns) + ["intercept"]))
                                    #  index = ["opinion"] + cu.features_join["control_short"]  + ["log_" + u for u in cu.features_join["control_short"]] + list(scores.columns) + ["intercept"]))

    beta_to_activation = (pd.Series(param_from_dict_params("opinion","activation",dict_params, significant_params)
                                    + param_from_dict_params("news_LT","activation", dict_params, significant_params)
                                    + param_from_dict_params("news_ST","activation", dict_params, significant_params)
                                    + param_from_dict_params("interaction","activation",dict_params, significant_params)
                                    + param_from_dict_params("activity","activation", dict_params, significant_params)
                                    + param_from_dict_params("intercept","activation", dict_params, significant_params),
                                    index = ["opinion"] + cu.features_join["norm_news_long"] + cu.features_join["norm_news_short"] + ["interaction_short"] + ["n_comments_short"] + ["intercept"]))
                                    # index = ["opinion"] + cu.features_join["norm_news_long"] + cu.features_join["norm_news_short"] + ["interaction_short"] + cu.features_join["control_short"] + ["log_" + u for u in cu.features_join["control_short"]] + ["intercept"]))
    
    opinions_users, beta_to_subreddit_ST, beta_to_activation, rotate_opinions = rotate_beta_opinions(opinions_users, beta_to_opinions, activation, beta_to_subreddit_ST, beta_to_activation)
    # if np.corrcoef(opinions_users, activation[:,0])[1,0] < 0:
    # # if dict_params["beta_opinion_to_activation_auto_loc"] < 0:
    #     beta_to_opinions = beta_to_opinions * (-1)
    #     beta_to_subreddit_ST.loc[scores.columns] = beta_to_subreddit_ST.loc[scores.columns] * (-1)
    #     beta_to_activation.loc["opinion"] = beta_to_activation.loc["opinion"] * (-1)
    #     opinions_users = opinions_users * (-1)

    return {"beta_to_opinions": beta_to_opinions,
            "beta_to_activity_ST": beta_to_activity_ST,
            "beta_to_subreddit_LT": beta_to_subreddit_LT,
            "beta_to_subreddit_ST": beta_to_subreddit_ST,
            "beta_to_interaction": beta_to_interaction,
            "beta_to_activation": beta_to_activation,
            "opinions_users": opinions_users,
            "opinions_median": opinions_median,
            "sociodemo_users": sociodemo_users,
            "activation": activation[:,0],
            "news_ST": news_ST, 
            "news_LT": news_LT,
            "interaction_ST": interaction_ST,
            "rotate_opinions": rotate_opinions
            }

def rotate_beta_opinions(opinions_users, beta_to_opinions, activation, beta_to_subreddit_ST, beta_to_activation):
    if np.corrcoef(opinions_users, activation[:,0])[1,0] < 0:
        rotate_opinions = True
        beta_to_opinions = beta_to_opinions * (-1)
        beta_to_subreddit_ST.loc[scores.columns] = beta_to_subreddit_ST.loc[scores.columns] * (-1)
        beta_to_activation.loc["opinion"] = beta_to_activation.loc["opinion"] * (-1)
        opinions_users = opinions_users * (-1)
    else:
        rotate_opinions = False
    return opinions_users, beta_to_subreddit_ST, beta_to_activation, rotate_opinions

def sample_posterior(guide, params, data, num_samples = 100):
    posterior_samples = guide.sample_posterior(PRNGKey(1000), params = params, sample_shape = (num_samples,))
    posterior_predictive = Predictive(model, posterior_samples, num_samples = num_samples)
    posterior_predictions = posterior_predictive(PRNGKey(1000), data, (None,None,None,None,None))
    
    all_posteriors = posterior_samples
    all_posteriors.update(posterior_predictions)
    (subreddit_LT, activity_LT, log_activity_LT, news_LT, subreddit_ST, activity_ST, log_activity_ST, news_ST, interaction_ST, activation, sociodemo_subreddit, popularity_subreddits, log_popularity_subreddits, subreddit_list) = data
    full_activity_LT = jnp.concat([activity_LT, log_activity_LT], axis = 1)[:,3][:,None]
    full_activity_ST = jnp.concat([activity_ST, log_activity_ST], axis = 1)[:,3][:,None]
    all_posteriors["opinions_pred"] = jnp.concat([(all_posteriors["sociodemo_users"][k] @ all_posteriors["beta_sociodemo_to_opinions"][k] + full_activity_LT @ all_posteriors["beta_activity_to_opinions"][k] + news_LT @ all_posteriors["beta_news_to_opinions"][k] + all_posteriors["beta_intercept_to_opinions"][k] + all_posteriors["opinions_noise"][k])[None,:] for k in range(num_samples)], axis  = 0)
    # all_posteriors["opinions_pred"] = jnp.concat([(all_posteriors["sociodemo_users"][k] @ all_posteriors["beta_sociodemo_to_opinions"][k] + full_activity_LT @ all_posteriors["beta_activity_to_opinions"][k] + news_LT @ all_posteriors["beta_news_to_opinions"][k] + all_posteriors["beta_intercept_to_opinions"][k] + all_posteriors["opinions_noise"][k])[None,:] for k in range(num_samples)], axis  = 0)
    return all_posteriors

def full_experiment(subreddit_class, n_epochs, lr, var_opinions = 0.1, multivariate = True, date = "240904", progress_bar = True, id = "001", print_loss = True, save = True, return_res = False, 
                    init_scale = 0.1, init_scale_opinions = 0.1, key = 1000, early_stop = True, early_stop_params = {"delta_loss": 100, "delta_params": 1.}):
    model_settings = {
    "lr": lr,
    "key": key,
    "n_epochs": n_epochs,
    "subreddit_class": subreddit_class,
    "var_opinions": var_opinions,
    "multivariate": multivariate,
    "init_scale": init_scale,
    "init_scale_opinions": init_scale_opinions
                  }
    df = load_dataframe(subreddit_class)
    data, data_obs = get_data(df, scores)
    
    init_params_normal, init_params_multivariatenormal = get_init_params(len(df))
    init_params = initialize_params(init_params_normal, init_params_multivariatenormal)
    guide, svi = get_guide(model, model_settings, data, data_obs,
                                        init_params_normal, init_params_multivariatenormal, 
                                        var_opinions = var_opinions, multivariate = multivariate,
                                        init_scale = init_scale, init_scale_opinions = init_scale_opinions)
    if save:
        path = path_to_exp + f"{subreddit_class}_{date}_{id}.pkl"
    else:
        path = None
    guide, svi_results, betas, model_settings = train(svi, guide, data, data_obs, model_settings, 
                                            init_params_normal, init_params_multivariatenormal, var_opinions = var_opinions, progress_bar = progress_bar, 
                                            path = path, print_loss = print_loss, key = key, early_stop = early_stop, early_stop_params = early_stop_params)
    
    if return_res:
        return guide, svi_results, betas, model_settings


def convert_pd_jnp_array(x):
    try:
        x1 = np.array([u.item() for u in x])
        if type(x) == pd.core.series.Series:
            return pd.Series(x1, index = x.index)
        else:
            return x1
    except:
        return x
    
def read_pkl(file):
    try:
        # return pd.read_pickle(file)
        with open(file, "rb") as f:
            data = dill.load(f)
        return data
    except:
        print(file)

def get_experiments_data(pickle, other_vars_settings = []):
    model_settings, betas,  = pickle["model_settings"], pickle["betas"]
    model_settings["loss"] = pickle["losses"][-1].item()
    setting_vars = ["subreddit_class", "lr", "var_opinions", "init_scale", "init_scale_opinions", "num_epochs_early_stop", "loss", "accuracy_mean", "accuracy_std", "time"] + other_vars_settings
    settings_s = pd.Series([model_settings[k] for k in setting_vars], index = setting_vars)
    opinions_s = betas["beta_to_opinions"][[u for u in betas["beta_to_opinions"].index if (u in cu.sociodemo_classes) or ("norm" in u)]].rename(lambda x: x + "_to_opinion")
    activation_s = betas["beta_to_activation"][:8].rename(lambda x: x + "_to_activation")
    interaction_s = betas["beta_to_interaction"][cu.sociodemo_classes].rename(lambda x: x + "_to_interaction")
    output = pd.concat([settings_s, opinions_s, activation_s, interaction_s])
    
    return output




if __name__ == "__main__":
    # 2 hours 40min
    for subreddit_class in ["activism", "skeptic", "action", "discussion"]:
        full_experiment(subreddit_class, 5000, 0.01, multivariate = True, date = "240905", id = "002", print_loss = False, progress_bar = False)
        print(subreddit_class)

        model_settings = {
        "lr": 0.01,
        "n_epochs": 5000,
        "subreddit_class": "activism",
        "var_opinions": 0.1,
                    }
        
        df = load_dataframe(subreddit_class)

        print("shortmedium")
        model_settings["robustness"] = "shortmedium"
        data, data_obs = get_data(create_df_merge_shortmedium(df), scores)
        subreddit_list = data[-1]
        init_params_normal, init_params_multivariatenormal = get_init_params(len(df))
        init_params = initialize_params(init_params_normal, init_params_multivariatenormal)
        guide, svi = get_guide(model, model_settings, data, data_obs,
                                            init_params_normal, init_params_multivariatenormal, multivariate = True)
        guide, svi_results, betas = train(svi, guide, data, data_obs, model_settings,
                                               init_params_normal, init_params_multivariatenormal,
                                                 progress_bar = True, path = path_to_exp + f"{subreddit_class}_shortmedium_240905_001.pkl", print_loss = True)

        print("longmedium")
        model_settings["robustness"] = "longmedium"
        data, data_obs = get_data(create_df_merge_longmedium(df), scores)
        subreddit_list = data[-1]
        init_params_normal, init_params_multivariatenormal = get_init_params(len(df))
        init_params = initialize_params(init_params_normal, init_params_multivariatenormal)
        guide, svi = get_guide(model, model_settings, data, data_obs,
                               init_params_normal, init_params_multivariatenormal, multivariate = True)
        
        guide, svi_results, betas = train(svi, guide, data, data_obs, model_settings,
                                          init_params_normal, init_params_multivariatenormal,
                                          progress_bar = True, path = path_to_exp + f"{subreddit_class}_longmedium_240905_001.pkl", print_loss = True)


