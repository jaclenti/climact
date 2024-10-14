import sys
sys.path += ["../src"]
import climact_shared.src.utils as cu
import causal_model
import pandas as pd
import numpy as np

def weighted_avg_cols(df, col1, col2, w1, w2):
    
    return (df[col1] * w1 + df[col2] * w2) / (w1 + w2)

def subreddits_union(df, col1, col2):
    return df[col1]|df[col2]


def create_df_merge_longmedium(df):
    subreddit_list = [u for u in sorted(cu.subreddits) if (f"r{u}_long" in df.columns)&(f"r{u}_medium" in df.columns)&(f"r{u}_short" in df.columns)]
    df_longmedium = pd.concat([
    df[cu.features_join["interaction_short"] + cu.features_join["norm_news_short"] + cu.features_join["control_short"] + [f"r{u}_short" for u in subreddit_list]],
    pd.DataFrame([weighted_avg_cols(df, u + "_medium", u + "_long", 4, 47) for u in (cu.features["norm_news"] + cu.features["control"])],
                 index = cu.features_join["norm_news_long"] + cu.features_join["control_long"]).T,
    pd.DataFrame([subreddits_union(df, f"r{u}_medium", f"r{u}_long") for u in subreddit_list], index = [f"r{u}_long" for u in subreddit_list]).T
                 ], axis = 1).assign(activation = df["activation"])
    return df_longmedium

def create_df_merge_shortmedium(df):
    subreddit_list = [u for u in sorted(cu.subreddits) if (f"r{u}_long" in df.columns)&(f"r{u}_medium" in df.columns)&(f"r{u}_short" in df.columns)]
    df_shortmedium = pd.concat([
    df[cu.features_join["interaction_long"] + cu.features_join["norm_news_long"] + cu.features_join["control_long"] + [f"r{u}_long" for u in subreddit_list]],
    pd.DataFrame([weighted_avg_cols(df, u + "_medium", u + "_short", 4, 1) for u in (cu.features["interaction"] + cu.features["norm_news"] + cu.features["control"])],
                 index = cu.features_join["interaction_short"] + cu.features_join["norm_news_short"] + cu.features_join["control_short"]).T,
    pd.DataFrame([subreddits_union(df, f"r{u}_medium", f"r{u}_short") for u in subreddit_list], index = [f"r{u}_short" for u in subreddit_list]).T
                 ], axis = 1).assign(activation = df["activation"])
    df_shortmedium[cu.features_join["interaction_short"]] = df_shortmedium[cu.features_join["interaction_short"]] > 0
    return df_shortmedium



def full_experiment(subreddit_class, n_epochs, lr, var_opinions = 0.1, multivariate = True, 
                    date = "240904", progress_bar = True, id = "001", print_loss = True, save = True, return_res = False, 
                    init_scale = 0.1, init_scale_opinions = 0.1, key = 1000, early_stop = True, 
                    early_stop_params = {"delta_loss": 100, "delta_params": 1.},
                    merge = None):
    model_settings = {
    "lr": lr,
    "key": key,
    "n_epochs": n_epochs,
    "subreddit_class": subreddit_class,
    "var_opinions": var_opinions,
    "multivariate": multivariate,
    "init_scale": init_scale,
    "init_scale_opinions": init_scale_opinions,
    "merge": merge
                  }
    df = causal_model.load_dataframe(subreddit_class)
    if merge == "shortmedium":
        df = create_df_merge_shortmedium(df)
    if merge == "longmedium":
        df = create_df_merge_longmedium(df)

    data, data_obs = causal_model.get_data(df, causal_model.scores)
    
    init_params_normal, init_params_multivariatenormal = causal_model.get_init_params(len(df))
    init_params = causal_model.initialize_params(init_params_normal, init_params_multivariatenormal)
    guide, svi = causal_model.get_guide(causal_model.model, model_settings, data, data_obs,
                                        init_params_normal, init_params_multivariatenormal, 
                                        var_opinions = var_opinions, multivariate = multivariate,
                                        init_scale = init_scale, init_scale_opinions = init_scale_opinions)
    if save:
        path = causal_model.path_to_exp + f"{subreddit_class}_{date}_{id}.pkl"
    else:
        path = None
    guide, svi_results, betas, model_settings = causal_model.train(svi, guide, data, data_obs, model_settings, 
                                            init_params_normal, init_params_multivariatenormal, var_opinions = var_opinions, progress_bar = progress_bar, 
                                            path = path, print_loss = print_loss, key = key, early_stop = early_stop, early_stop_params = early_stop_params)
    
    if return_res:
        return guide, svi_results, betas, model_settings


if __name__ == "__main__":
    _, subreddit_class, n_epochs, lr, var_opinions, date, init_scale, init_scale_opinions, key, merge = sys.argv
    n_epochs, lr, var_opinions, init_scale, init_scale_opinions, key = int(n_epochs), int(lr), int(var_opinions), int(init_scale), int(init_scale_opinions), int(key) 
    lr = lr / 1000
    var_opinions = var_opinions / 1000 if var_opinions > 0 else 1e-9
    
    init_scale = init_scale / 1000
    init_scale_opinions = init_scale_opinions / 10000
    print(merge, var_opinions, key)
    
    full_experiment(subreddit_class, n_epochs, lr, 
                                var_opinions = var_opinions, 
                                multivariate = True, 
                                key = key,
                                date = date, 
                                progress_bar = False, 
                                id = f"robustness_{merge}_v{var_opinions}_k{key}", 
                                print_loss = False, save = True,
                                return_res = False, init_scale = init_scale, 
                                init_scale_opinions = init_scale_opinions, 
                                merge = merge)
            