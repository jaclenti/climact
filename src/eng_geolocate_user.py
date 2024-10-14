import pandas as pd 
import climact_shared.src.utils as cu


subreddits_country = pd.read_csv(cu.data_path + "US_CA_subreddit_geolocation.csv", index_col = 0)

def geolocate_from_users_df(df):
    df_admin_users =  df.merge(subreddits_country)[["author", "subreddit", "admin1code"]].groupby(["author", "subreddit"]).first().reset_index()
    df_admin_users = df_admin_users[df_admin_users["author"].isin(df_admin_users.groupby("author").count().reset_index().query("subreddit == 1").author)][["author", "admin1code"]]

    return df_admin_users






