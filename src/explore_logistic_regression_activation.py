import sys
sys.path += ["../src"]
import features_cox_week as ft
import climact_shared.src.utils as cu
import climact_shared.src.explore_cox_logreg_experiments as cle
import numpy as np
from importlib import reload
import pandas as pd
from glob import glob
import re
import climact_shared.src.explore_cox_logreg_experiments as cle

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from climact_shared.src.eng_create_features_df import create_features_df

from sklearn import linear_model
from scipy import stats


class LogisticRegression_activation():
    def __init__(self, subreddit_class, months_list = cu.all_months[:-2]):
        self.subreddit_class = subreddit_class
        self.months_list = months_list

    def load_data(self, balanced_classes = True, random_state = 100, verbose = False):
        self.df = create_features_df(self.subreddit_class, self.months_list, 
                                     balanced_classes = balanced_classes, 
                                     random_state = random_state,
                                     verbose = verbose)
        
    def repeated_bootstrap(self, n_samples = 10, random_state = 1, select_features = None, n_components = 30, select_time = False, test = False):
        coefs = []
        for k in range(n_samples):
            coefs.append(self.logistic_regression(n_components = n_components, bootstrap = True, test = test, select_time = select_time,
                                                  select_features = select_features, random_state = k * random_state))
        if test:
            bootstrap_df = pd.DataFrame([u[0] for u in coefs])
            exp_df = cle.exp_df_from_bootstrap(bootstrap_df)
            test_performances = pd.DataFrame([cle.confusion_matrix_analysis(u[1]) for u in coefs])
            summary_test = pd.DataFrame([test_performances.mean(), test_performances.std()], index = ["mean", "std"]).T
            return exp_df, summary_test
        else:
            bootstrap_df = pd.DataFrame(coefs)
            exp_df = cle.exp_df_from_bootstrap(bootstrap_df)
            return exp_df

    
    def logistic_regression(self, 
                            n_components = 30, 
                            bootstrap = False, 
                            random_state = None,
                            select_features = None,
                            select_time = False,
                            test = False):
        if bootstrap:
            df = cle.get_bootstrap_df(self.df.copy(), random_state = random_state)
        else:
            df = self.df.copy()
        
        if test:
            msk = np.random.rand(len(df)) < 0.75
            data_train = df.drop("duration", axis = 1)[msk]
            data_test = df.drop("duration", axis = 1)[~msk]
            X, y = data_train.drop("activation", axis = 1), data_train["activation"]
            X_test, y_test = data_test.drop("activation", axis = 1), data_test["activation"]
        else:
            data_train = df.drop("duration", axis = 1)
            X, y = data_train.drop("activation", axis = 1), data_train["activation"]
        if select_features:
            selected_cols = []
            selected_cols = [u for u in self.df.columns for features_group in [u for u in select_features if u != "subreddit"] for feature in cu.features[features_group] if re.match(feature, u)]
            if "subreddit" in select_features:
                selected_cols += [u for u in self.df.columns if u[0] == "r"]
            selected_cols = list(set(selected_cols))
            if select_time:
                selected_cols = [u for u in selected_cols if u[-len(select_time):] == select_time]
            X = X[selected_cols]
            if test:
                X_test = X_test[selected_cols]
        
        scaler = StandardScaler()
        scaler.fit(X)
        X_scale = scaler.transform(X)
        
        if n_components > X.shape[1]:
            n_components = X.shape[1]
        pca = PCA(n_components = n_components)
        pca.fit(X_scale)
        X_pca = pca.transform(X_scale)
        # X_pca = X_scale.copy()

        logistic = LogisticRegression(max_iter=10000, tol=0.1)
        logistic.fit(X_pca, y)
        
        original_coef = np.dot(logistic.coef_, pca.components_)[0]
        if test:
            X_scale_test = scaler.transform(X_test)
            X_pca_test = pca.transform(X_scale_test)
            cf = pd.DataFrame([logistic.predict(X_pca_test) + 0., y_test + 0.], index = ["predict", "actual"]).T.value_counts().unstack()
            return pd.Series(original_coef, index = X.columns), cf
        else:
            return pd.Series(original_coef, index = X.columns)
        

class separate_PCA():
    def __init__(self, n_components_dict = {"subreddit_long": 20,
                                       "control_long": 3,
                                       "control_short": 3,
                                       "interaction_short": 8,
                                       "norm_news_long": 3,
                                       "sociodemo": 4},
                                       df = pd.DataFrame([])):
        self.n_components_dict = n_components_dict
        self.pca = {}
        self.cols = {}
        self.n_components_list = [(features, n_components_dict[features]) for features in n_components_dict]
        
        for features, n_components in self.n_components_list:
            self.pca[features] = PCA(n_components = n_components)
            self.cols[features] = [u for u in cu.features_join[features] if u in df.columns]
        self.df = df

    
    def fit(self, X):
        X_df = pd.DataFrame(X, columns = self.df.columns)
        for features, n_components in self.n_components_list:
            self.pca[features].fit(X_df[self.cols[features]])
            
    def transform(self, X):
        X_pca_list = []
        X_df = pd.DataFrame(X, columns = self.df.columns)
        for features, n_components in self.n_components_list:
            X_pca_list.append(self.pca[features].transform(X_df[self.cols[features]]))
        return np.array(np.concatenate(X_pca_list, axis = 1))

    def inverse_transform(self, coef_):
        coefs = []
        j = 0
        for features, n_components in self.n_components_list:
            coefs.append(pd.Series(np.dot(coef_[:, j:(j + n_components)], 
                                          self.pca[features].components_)[0], 
                                          index = self.cols[features]))
            j += n_components
        return pd.concat(coefs)
        
        
        
class LogisticRegression_separate_pca():
    def __init__(self, subreddit_class, months_list = cu.all_months[:-2]):
        self.subreddit_class = subreddit_class
        self.months_list = months_list

    def load_data(self, balanced_classes = True, random_state = 100, verbose = False):
        self.df = create_features_df(self.subreddit_class, self.months_list, 
                                     balanced_classes = balanced_classes, 
                                     random_state = random_state,
                                     verbose = verbose)
        
    def repeated_bootstrap(self, n_samples = 10, random_state = 1, select_features = None, 
                           n_components_dict = {"subreddit_long": 50,
                                            "control_long": 3,
                                            "control_short": 3,
                                            "interaction_short": 4,
                                            "norm_news_long": 3}, 
                           select_time = False, test = False):
        coefs = []
        for k in range(n_samples):
            coefs.append(self.logistic_regression(n_components_dict = n_components_dict, bootstrap = True, test = test, select_time = select_time,
                                                  select_features = select_features, random_state = k * random_state))
        if test:
            bootstrap_df = pd.DataFrame([u[0] for u in coefs])
            exp_df = cle.exp_df_from_bootstrap(bootstrap_df)
            test_performances = pd.DataFrame([cle.confusion_matrix_analysis(u[1]) for u in coefs])
            summary_test = pd.DataFrame([test_performances.mean(), test_performances.std()], index = ["mean", "std"]).T
            return exp_df, summary_test
        else:
            bootstrap_df = pd.DataFrame(coefs)
            exp_df = cle.exp_df_from_bootstrap(bootstrap_df)
            return exp_df

    
    def logistic_regression(self, 
                            n_components_dict = {"subreddit_long": 20,
                                            "control_long": 3,
                                            "control_short": 3,
                                            "interaction_short": 8,
                                            "norm_news_long": 3}, 
                            bootstrap = False, 
                            random_state = None,
                            select_features = None,
                            select_time = False,
                            test = False):
        if bootstrap:
            df = cle.get_bootstrap_df(self.df.copy(), random_state = random_state)
        else:
            df = self.df.copy()
        
        if test:
            msk = np.random.rand(len(df)) < 0.75
            data_train = df.drop("duration", axis = 1)[msk]
            data_test = df.drop("duration", axis = 1)[~msk]
            X, y = data_train.drop("activation", axis = 1), data_train["activation"]
            X_test, y_test = data_test.drop("activation", axis = 1), data_test["activation"]
        else:
            data_train = df.drop("duration", axis = 1)
            X, y = data_train.drop("activation", axis = 1), data_train["activation"]
        all_cols = [u for k in n_components_dict for u in cu.features_join[k] if u in df.columns]
        selected_cols = [u for u in df.columns if u in all_cols]
        # selected_cols = [u for u in self.df.columns for features_group in [u for u in select_features if u != "subreddit"] for feature in ft.features[features_group] if re.match(feature, u)]
        # if "subreddit" in select_features:
        #     selected_cols += [u for u in self.df.columns if u[0] == "r"]
        # selected_cols = list(set(selected_cols))            
        X = X[selected_cols]
        if test:
            X_test = X_test[selected_cols]
        
        scaler = StandardScaler()
        scaler.fit(X)
        X_scale = scaler.transform(X)
        
        spca = separate_PCA(n_components_dict = n_components_dict, df = df[selected_cols])
        spca.fit(X_scale)
        X_pca = spca.transform(X_scale)
        
        logistic = LogisticRegression(max_iter=10000, tol=0.1)
        logistic.fit(X_pca, y)
        original_coef = spca.inverse_transform(logistic.coef_)
        
        if test:
            X_scale_test = scaler.transform(X_test)
            X_pca_test = spca.transform(X_scale_test)
            cf = pd.DataFrame([logistic.predict(X_pca_test) + 0., y_test + 0.], index = ["predict", "actual"]).T.value_counts().unstack()
            return pd.Series(original_coef), cf
        else:
            return pd.Series(original_coef)




def model_selection(df_, k = None,
                    separate_pca = False,
                    n_components_dict = {
                        "control_long": 3,
                        "control_short": 3,
                        "interaction_short": 8,
                        "norm_news_long": 3}):
    df = df_.drop("duration", axis = 1)
    msk = np.random.rand(len(df)) < 0.75
    data_train = df[msk]
    data_test = df[~msk]
    X, y = data_train.drop("activation", axis = 1), data_train["activation"]
    X_test, y_test = data_test.drop("activation", axis = 1), data_test["activation"]    
    
    if separate_pca:
        if n_components_dict is None:
            n_components_dict = {f"{feat}_{p}": len(cu.features[feat]) 
                                 for feat in cu.features
                                 for p in cu.periods if feat not in ["target", "sociodemo", "tot_news"]}
            n_components_dict.update({f"subreddit_{p}": k for p in ["short", "medium", "long", "short_long_ratio"]})
            n_components_dict.update({"sociodemo": 8})
        all_cols = [u for k in n_components_dict for u in cu.features_join[k] if u in df.columns]
        selected_cols = [u for u in X.columns if u in all_cols]
        k = np.array(list(n_components_dict.values())).sum()
        pca = separate_PCA(n_components_dict = n_components_dict, df = df[selected_cols])
    else: 
        pca = PCA(n_components = k)
        selected_cols = X.columns
    
    scaler = StandardScaler()
    scaler.fit(X[selected_cols])
    X_scale = scaler.transform(X[selected_cols])
    
    pca.fit(X_scale)
    X_pca = pca.transform(X_scale)

    logistic = LogisticRegression(max_iter=10000, tol=0.1)
    logistic.fit(X_pca, y)
    
    beta = logistic.coef_
    yi = np.array(y)
    Xi = X_pca
    yi_hat = (Xi @ beta.T)[:,0]
    log_likelihood =np.sum(-np.log(1 + np.exp(yi_hat)) + yi * yi_hat)

    N = len(y)

    X_scale_test = scaler.transform(X_test[selected_cols])
    X_pca_test = pca.transform(X_scale_test)
    accuracy = np.mean(logistic.predict(X_pca_test) == y_test)
            
    return {"k": k,
            "accuracy": accuracy,
            "log-likelihood": log_likelihood,
            "AIC": cu.compute_aic(N, log_likelihood, k), 
            "BIC": cu.compute_bic(N, log_likelihood, k)}
