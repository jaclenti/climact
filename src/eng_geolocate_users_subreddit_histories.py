
import pandas as pd 
import sys
sys.path += ["../src"]
sys.path += ["../src/geonameslocator"]
import geolocator
import json
from glob import glob
import numpy as np
import climact_shared.src.utils as cu

import spark_init
sc = spark_init.spark_context()

def resilient_json(s):
    try:
        return json.loads(s)
    except:
        return {}
    

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
type_txts = ["RS", "RC"]

sc.addPyFile("../src/geonameslocator/geolocator.py")


def exist_geolocation(s):
    exist = geolocator.find_best_match(s) is not None
    return exist


if __name__ == '__main__':
    for activation in ["weak"]:
        for type_txt in type_txts:
            for folder in sorted(glob(cu.data_path + f"history_{activation}_authors/{type_txt}*")):
                if len(glob(folder + "/*")) > 0:
                    folder_id = activation + "_" + folder.split("/")[-1]
                    print(folder_id)
                    df_geo = pd.DataFrame(sc.textFile(folder + "/*bz2").map(lambda x: resilient_json(x)).collect(), columns = cu.columns).dropna(subset = "subreddit")\
                                                .loc[lambda x: [exist_geolocation(u) for u in x["subreddit"]]][["author", "subreddit"]].value_counts()
                    
                    
                    df_geo = pd.DataFrame(sc.textFile(folder + "/*bz2").map(lambda x: resilient_json(x)).collect(), columns = cu.columns).dropna(subset = "subreddit")\
                    .assign(geolocation = lambda x: [geolocator.find_best_match(u) for u in x["subreddit"]]).dropna(subset = "geolocation")[["author", "subreddit", "geolocation"]].value_counts()
                                                #.loc[lambda x: [exist_geolocation(u) for u in x["subreddit"]]][["author", "subreddit", "geolocation"]].value_counts()
                    df_geo.to_csv(cu.data_path + f"geolocated_comments_users_subreddit/{folder_id}.csv")

        