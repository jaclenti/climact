import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
sys.path += ["../src"]
import causal_model
from matplotlib.pyplot import subplots as sbp
from importlib import reload
from glob import glob
import climact_shared.src.utils as cu

path_to_data = "/data/shared/xxx/climact/experiments/"



if __name__ == '__main__':
    for subreddit_class in cu.subreddit_classes:
        for i,var_opinions in enumerate(np.array([1e-9, 0.001, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 1., 10., 100.])):
        # for i,var_opinions in enumerate(np.append(10e-9, np.logspace(-3, 2, 11))):
            print(i, subreddit_class, var_opinions)
            var_opinions = round(var_opinions, 10)
            guide, svi_results, betas = causal_model.full_experiment(subreddit_class, 
                                                                     5000, 
                                                                     lr = 0.01, 
                                                                     var_opinions = var_opinions, 
                                                                     multivariate = True, 
                                                                     date = "240927",
                                                                     progress_bar = False, 
                                                                     id = f"{(i + 0 * 100):03d}", 
                                                                     print_loss = False, 
                                                                     save = True, 
                                                                     return_res = True,
                                                                     init_scale = 0.1,
                                                                     init_scale_opinions = 0.001)
            