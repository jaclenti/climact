import sys
sys.path += ["../src"]
import numpy as np

import causal_model
from causal_model import load_dataframe, get_data, scores, get_init_params, initialize_params, path_to_exp, params_list
import jax, jaxlib
import jax.numpy as jnp
import jax.random as random
from jax.random import PRNGKey
from scipy.special import expit as np_sigmoid
import numpyro
from time import time
from jax.scipy.special import expit as sigmoid
from numpyro.infer import SVI, Trace_ELBO, TraceGraph_ELBO, TraceMeanField_ELBO, MCMC, NUTS, Predictive, init_to_value
from numpyro.infer.autoguide import AutoNormal, AutoMultivariateNormal, AutoGuideList, AutoLowRankMultivariateNormal
from numpyro import distributions
from numpyro.optim import Adam
import dill

all_vars = {"sociodemo_to_opinions": True, 
             "activity_to_opinions": True, 
             "news_to_opinions": True, 
             "interaction_to_activation": True, 
             "opinion_to_activation": True,
             "activity_to_sub_LT": True,
             "activity_to_sub_ST": True,
             "activity_to_interaction": True,
             "activity_to_activation": True, 
             "news_LT_to_activation": True, 
             "news_ST_to_activation": True, 
             "intercept_to_activation": True}

def get_guide(model, model_settings, data, data_obs, with_vars, init_params_normal, init_params_multivariatenormal, var_opinions = 0.1, multivariate = True, init_scale = 0.1, init_scale_opinions = 0.1):
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
    svi.init(PRNGKey(100), data, data_obs, with_vars, var_opinions)
    
    return guide, svi

def model(data, data_obs, with_vars,  var_opinions = 0.1):
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

    beta_sociodemo_to_opinions = numpyro.sample("beta_sociodemo_to_opinions", distributions.MultivariateNormal(jnp.zeros(socio_dim), covariance_matrix = jnp.eye(socio_dim,socio_dim))) * with_vars["sociodemo_to_opinions"]
    beta_activity_to_opinions = numpyro.sample("beta_activity_to_opinions", distributions.MultivariateNormal(jnp.zeros((act_dim)), jnp.eye(act_dim,act_dim))) * with_vars["activity_to_opinions"]
    beta_news_to_opinions = numpyro.sample("beta_news_to_opinions", distributions.MultivariateNormal(jnp.zeros((news_dim)), jnp.eye(news_dim,news_dim))) * with_vars["news_to_opinions"]
    beta_intercept_to_opinions = numpyro.sample("beta_intercept_to_opinions", distributions.Normal(jnp.array([0.])))
    
    opinions_median = (sociodemo_users @ beta_sociodemo_to_opinions + full_activity_LT @ beta_activity_to_opinions + news_LT @ beta_news_to_opinions)[:,None]
    opinions_noise = numpyro.sample("opinions_noise", distributions.Normal(jnp.zeros((n_authors)), np.sqrt(var_opinions) * jnp.ones((n_authors))))[:,None]
    opinions = opinions_median + opinions_noise
    opinions = (opinions - opinions.mean()) / opinions.std()
    
    beta_sociodemo_to_sub_LT = numpyro.sample("beta_sociodemo_to_sub_LT", distributions.MultivariateNormal(jnp.zeros((socio_dim)), jnp.eye(socio_dim,socio_dim)))
    beta_activity_to_sub_LT = numpyro.sample("beta_activity_to_sub_LT", distributions.MultivariateNormal(jnp.zeros((act_dim)), jnp.eye(act_dim,act_dim))) * with_vars["activity_to_sub_LT"]
    beta_popularity_to_sub_LT = numpyro.sample("beta_popularity_to_sub_LT", distributions.Normal(jnp.zeros((pop_dim))))
    beta_intercept_to_sub_LT = numpyro.sample("beta_intercept_to_sub_LT", distributions.Normal(jnp.array([0.])))

    logits_subreddit_LT = beta_sociodemo_to_sub_LT * sociodemo_users @ sociodemo_subreddit.T + full_popularity_subreddits @ beta_popularity_to_sub_LT + (full_activity_LT @ beta_activity_to_sub_LT)[:,None] +  beta_intercept_to_sub_LT
        
    beta_activity_to_activity_ST = numpyro.sample("beta_activity_to_activity_ST", distributions.Normal(jnp.zeros((act_dim)))) 
    beta_intercept_to_activity_ST = numpyro.sample("beta_intercept_to_activity_ST", distributions.Normal(jnp.zeros((act_dim))))
    
    median_activity_ST = beta_activity_to_activity_ST * activity_LT + beta_intercept_to_activity_ST
    
    beta_retention_to_sub_ST = numpyro.sample("beta_retention_to_sub_ST", distributions.Normal(jnp.zeros(1)))
    beta_sociodemo_to_sub_ST = numpyro.sample("beta_sociodemo_to_sub_ST", distributions.MultivariateNormal(jnp.zeros((socio_dim)), jnp.eye(socio_dim,socio_dim)))
    beta_activity_to_sub_ST = numpyro.sample("beta_activity_to_sub_ST", distributions.MultivariateNormal(jnp.zeros((act_dim)), jnp.eye(act_dim,act_dim))) * with_vars["activity_to_sub_ST"]
    beta_popularity_to_sub_ST = numpyro.sample("beta_popularity_to_sub_ST", distributions.Normal(jnp.zeros((pop_dim))))
    beta_intercept_to_sub_ST = numpyro.sample("beta_intercept_to_sub_ST", distributions.Normal(jnp.array([0.])))
    
    logits_subreddit_ST = subreddit_LT * beta_retention_to_sub_ST + beta_sociodemo_to_sub_ST * opinions @ sociodemo_subreddit.T  + (full_activity_ST @ beta_activity_to_sub_ST)[:,None] + (full_popularity_subreddits @ beta_popularity_to_sub_ST)[None,:] + beta_intercept_to_sub_ST
    
    beta_opinion_to_interaction = numpyro.sample("beta_opinion_to_interaction", distributions.Normal(jnp.array([0.])))
    beta_activity_to_interaction = numpyro.sample("beta_activity_to_interaction", distributions.MultivariateNormal(jnp.zeros((act_dim)), jnp.eye(act_dim,act_dim))) * with_vars["activity_to_interaction"]
    beta_sociodemo_to_interaction = numpyro.sample("beta_sociodemo_to_interaction", distributions.Normal(jnp.zeros(socio_dim)))
    beta_intercept_to_interaction = numpyro.sample("beta_intercept_to_interaction", distributions.Normal(jnp.array([0.])))
    
    logits_interaction_ST = (full_activity_ST @ beta_activity_to_interaction)[:,None] + (subreddit_ST @ sociodemo_subreddit @ beta_sociodemo_to_interaction[:,None]) + beta_intercept_to_interaction
    # logits_interaction_ST = (beta_opinion_to_interaction * opinions) + (activity_ST @ beta_activity_to_interaction)[:,None] + (log_activity_ST @ beta_log_activity_to_interaction)[:,None] + (subreddit_ST @ sociodemo_subreddit @ beta_sociodemo_to_interaction[:,None]) + beta_intercept_to_interaction
     
    beta_interaction_to_activation = numpyro.sample("beta_interaction_to_activation", distributions.Normal(jnp.zeros((1)))) * with_vars["interaction_to_activation"]
    beta_opinion_to_activation = numpyro.sample("beta_opinion_to_activation", distributions.Normal(jnp.array([0.]))) * with_vars["opinion_to_activation"]
    beta_activity_to_activation = numpyro.sample("beta_activity_to_activation", distributions.MultivariateNormal(jnp.zeros((act_dim)), jnp.eye(act_dim,act_dim))) * with_vars["activity_to_activation"]
    beta_news_LT_to_activation = numpyro.sample("beta_news_LT_to_activation", distributions.MultivariateNormal(jnp.zeros((news_dim)), jnp.eye(news_dim,news_dim))) * with_vars["news_LT_to_activation"]
    beta_news_ST_to_activation = numpyro.sample("beta_news_ST_to_activation", distributions.MultivariateNormal(jnp.zeros((news_dim)), jnp.eye(news_dim,news_dim))) * with_vars["news_ST_to_activation"]
    beta_intercept_to_activation = numpyro.sample("beta_intercept_to_activation", distributions.Normal(jnp.array((0.)))) * with_vars["intercept_to_activation"]
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

def train(svi, guide, data, data_obs, with_vars, model_settings, init_params_normal, init_params_multivariatenormal, var_opinions = 0.1, progress_bar = False, path = None, 
          print_loss = True, early_stop = True, key = 1000, early_stop_params = {"delta_loss": 100, "delta_params": 1.}):
    key = np.random.randint(low = 1, high = 1000)
    t0 = time()
    svi.run_early_stop = causal_model.run_early_stop.__get__(svi)
    if early_stop:
        svi_results, losses = svi.run_early_stop(random.PRNGKey(key), model_settings["n_epochs"], data = data, data_obs = data_obs, with_vars = with_vars,
                                                 var_opinions = var_opinions, delta_loss = early_stop_params["delta_loss"], delta_params = early_stop_params["delta_params"])
    else:                                             
        svi_results = svi.run(random.PRNGKey(key), model_settings["n_epochs"], data = data, data_obs = data_obs, with_vars = with_vars, var_opinions = var_opinions, progress_bar = progress_bar)
        losses = svi_results.losses
    t1 = time()
    model_settings["time"] = t1 - t0
    model_settings["num_epochs_early_stop"] = len(losses)
    model_settings["loss"] = losses[-1].item()
    accuracy_mean, accuracy_std = causal_model.get_accuracy(guide, svi_results, data = data)
    model_settings["accuracy_mean"], model_settings["accuracy_std"] = accuracy_mean, accuracy_std
    betas = causal_model.get_trained_beta(svi_results.params, data, data_obs, init_params_normal, init_params_multivariatenormal)
    # if print_loss:
    #     print(losses[-1])
    #     fig, axes = sbp(ncols = 2, figsize = (4,2))
    #     axes[0].plot(losses)
    #     axes[1].plot(jnp.abs(jnp.diff(losses)))
    #     axes[1].set_yscale("log")
    if path:
        causal_model.save_model(guide, svi_results, model_settings, path, betas)

    return guide, svi_results, betas, model_settings



def full_experiment(subreddit_class, n_epochs, lr, var_opinions = 0.1, multivariate = True, date = "240904", progress_bar = True, 
                    id = "001", print_loss = True, save = True, return_res = False, init_scale = 0.1, init_scale_opinions = 0.1, key = None,
                    with_vars = all_vars, ablation = ""):
    if key == None:
        key = np.random.randint(low = 0, high = 10000)
    model_settings = {
    "lr": lr,
    "n_epochs": n_epochs,
    "subreddit_class": "activism",
    "var_opinions": var_opinions,
    "multivariate": multivariate,
    "init_scale": init_scale,
    "init_scale_opinions": init_scale_opinions,
    "key":key,
    "ablation": ablation
                  }
    df = load_dataframe(subreddit_class)
    data, data_obs = get_data(df, scores)
    

    init_params_normal, init_params_multivariatenormal = get_init_params(len(df))
    init_params = initialize_params(init_params_normal, init_params_multivariatenormal)
    guide, svi = get_guide(model, model_settings, data, data_obs, with_vars,
                                        init_params_normal, init_params_multivariatenormal, 
                                        var_opinions = var_opinions, multivariate = multivariate,
                                        init_scale = init_scale, init_scale_opinions = init_scale_opinions,)
    if save:
        path = path_to_exp + f"{subreddit_class}_{date}_{id}.pkl"
    else:
        path = None
    guide, svi_results, betas, model_settings = train(svi, guide, data, data_obs, with_vars, model_settings, 
                                            init_params_normal, init_params_multivariatenormal, var_opinions = var_opinions, progress_bar = progress_bar, 
                                            path = path, print_loss = print_loss)
    if return_res:
        return guide, svi_results, betas, model_settings



if __name__ == "__main__":
    _, subreddit_class, n_epochs, lr, var_opinions, date, init_scale, init_scale_opinions, key, ablation = sys.argv
    n_epochs, lr, var_opinions, init_scale, init_scale_opinions, key = int(n_epochs), int(lr), int(var_opinions), int(init_scale), int(init_scale_opinions), int(key) 
    lr = lr / 1000
    var_opinions = var_opinions / 1000 if var_opinions > 0 else 1e-9
    
    init_scale = init_scale / 1000
    init_scale_opinions = init_scale_opinions / 10000
    
    
    if ablation == "complete":
        vars_ablation = all_vars.copy()

    if ablation == "no_sociodemo":
        vars_ablation = all_vars.copy()
        vars_ablation["sociodemo_to_opinions"] = False
    
    if ablation == "no_news":
        vars_ablation = all_vars.copy()
        vars_ablation["news_to_opinions"] = False
        vars_ablation["news_LT_to_activation"] = False
        vars_ablation["news_ST_to_activation"] = False
        
    if ablation == "no_interactions":
        vars_ablation = all_vars.copy()
        vars_ablation["interaction_to_activation"] = False

    if ablation == "no_activity":
        vars_ablation = all_vars.copy()
        vars_ablation["activity_to_sub_LT"] = False
        vars_ablation["activity_to_sub_ST"] = False
        vars_ablation["activity_to_interaction"] = False
        vars_ablation["activity_to_activation"] = False

    if ablation not in ["complete", "no_sociodemo", "no_news", "no_interactions", "no_activity"]:
        print(ablation, "Wrong ablation")
    
    print(var_opinions, key, ablation)
    full_experiment(subreddit_class, n_epochs, lr, 
                                var_opinions = var_opinions, 
                                multivariate = True, 
                                key = key,
                                date = date, 
                                progress_bar = False, 
                                id = f"abl_{ablation}_v{var_opinions}_k{key}", 
                                print_loss = False, save = True,
                                return_res = False, init_scale = init_scale, 
                                init_scale_opinions = init_scale_opinions, 
                                with_vars = vars_ablation,
                                ablation = ablation)
            