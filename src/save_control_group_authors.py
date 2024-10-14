import numpy as np
import pandas as pd
import sys

from datetime import datetime, date
sys.path += ['../src/']
import sys
import json
sys.path += ['../src/']
from glob import glob
import spark_init
sc = spark_init.spark_context()

import json

def resilient_json(s):
    try:
        return json.loads(s)
    except:
        return {}


if __name__ == '__main__':
    # data_strong_RC = sc.textFile("/data/big/xxx/climact/data/strong_activation_comments/*/part-*.bz2").map(resilient_json)
    # data_strong_RS = sc.textFile("/data/big/xxx/climact/data/strong_activation_submissions/*/part-*.bz2").map(resilient_json)
    # # score is the sum of upvote - downvote
    # columns = ["author", "subreddit", "body", "created_utc", "id", "link_id", "score", "subreddit_id", "parent_id"]
    # df_strong_RC = pd.DataFrame(data_strong_RC.collect(), columns = columns)
    # df_strong_RS = pd.DataFrame(data_strong_RS.collect(), columns = columns)
    # df_strong = pd.concat([df_strong_RC.assign(type_txt = "RC"), df_strong_RS.assign(type_txt = "RS")])
    # # convert created_utc in pandas date
    # df_strong["date"] = pd.to_datetime(df_strong["created_utc"], unit = "s")
    # # remove all the bots
    # bot_list=list(pd.read_csv('/data/big/xxx/climact/data/bot_list.csv')['author'])
    # df_strong = df_strong[~df_strong['author'].isin(bot_list)]
    # # remove all the comments/submissions with negative score (because they could be "against" their subreddit)
    # df_strong = df_strong.query("score > 0")
    # # keep only the post from 2013, so we can track back their histories
    # # for each year/month count the different authors we have
    # # so we can sample the control group with the same number of authors
    # author_activation_date = df_strong[df_strong["date"] > datetime(2013, 1, 1)].sort_values("date").groupby("author").first().sort_values("date")[["type_txt", "date"]].reset_index()
    # author_activation_date = author_activation_date.assign(year = lambda x: x.date.dt.year, month = lambda x: x.date.dt.month)
    # # I add all the zero entries for simplifying the sample later
    # count_activation_month = author_activation_date.groupby(["year", "month", "type_txt"]).count()["author"].unstack().fillna(0).stack()
    
    count_activation_month = pd.read_pickle(f"/data/big/xxx/climact/data/count_strong_activation_month.pkl").unstack()


    for year in count_activation_month.reset_index().year.unique():
        print(year)
        for month in count_activation_month.reset_index().month.unique():
            print(month)
            for type_txt in ["RC", "RS"]:
                number_authors = int(count_activation_month.loc[year, month, type_txt])
                    
                if number_authors > 0:
                    
                    type_data = "comments" if type_txt == "RC" else "submissions"
                    data = sc.textFile(f'/data/shared/reddit/{type_data}/{year}/{type_txt}_{year}-{month:02d}.bz2').map(resilient_json)
                    data = data.filter(lambda x: x.get('author') not in bot_list)

                    authors_one_month = data.map(lambda x: (x.get('author'),f'{month:02d}_{year}'))
                    

                    # I sample more authors, so I remove the duplicates later
                    sample_control_authors = authors_one_month.takeSample(False, number_authors * 3)
                    df_sample_authors = pd.DataFrame(sample_control_authors,columns=['author','month'])
                
                    #remove duplicates
                    df_sample_authors=df_sample_authors.drop_duplicates(subset=["author"],keep='first')
                    
                    # remove authors in my dataset
                    # I don't do it because I could be interested in a completely random sample of authors
                    # maybe some of them will activate some years later...
                    # df_sample_authors = df_sample_authors[-df_sample_authors['author'].isin(activated_authors)]
                    df_sample_authors = df_sample_authors.sample(n = number_authors)
                    df_sample_authors.to_pickle(f'/data/big/xxx/climact/data/strong_activation_{type_data}_control_authors/{year}_{month:02d}_{type_txt}.pickle')