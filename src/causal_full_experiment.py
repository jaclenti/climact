import sys
import causal_model
from time import time

if __name__ == "__main__":
    _, subreddit_class, n_epochs, lr, var_opinions, date, init_scale, init_scale_opinions, key =  sys.argv
    n_epochs, lr, var_opinions, init_scale, init_scale_opinions, key = int(n_epochs), int(lr), int(var_opinions), int(init_scale), int(init_scale_opinions), int(key)
    id = f"{var_opinions:02d}_{init_scale:02d}_{init_scale_opinions:03d}_{lr}_k{key}"
    lr = lr / 1000
    var_opinions = var_opinions / 1000 if var_opinions > 0 else 1e-9
    
    init_scale = init_scale / 1000
    init_scale_opinions = init_scale_opinions / 10000
    print(subreddit_class, "lr", lr, "var_opinions", var_opinions, "init_scale", init_scale, "init_scale_opinions", init_scale_opinions)
    t0 = time()

    guide, svi_results, betas, model_settings = causal_model.full_experiment(subreddit_class = subreddit_class, 
                                                             n_epochs = n_epochs, 
                                                             lr = lr, 
                                                             var_opinions = var_opinions, 
                                                             multivariate = True, 
                                                             date = date, 
                                                             progress_bar = False, 
                                                             id = id, 
                                                             key = key,
                                                             print_loss = False, 
                                                             save = True, 
                                                             return_res = True, 
                                                             init_scale = init_scale, 
                                                             init_scale_opinions = init_scale_opinions, 
                                                             early_stop = True,
                                                             early_stop_params = {"delta_loss": 50, "delta_params": .1})
    
    t1 = time()
    print(round(t1 - t0, 2))