import sys
import json
sys.path += ['../src/']
import os, shutil
import subprocess
from glob import glob
import spark_init
sc = spark_init.spark_context()



# Method for "unpack" the tuple and write the useful keys values as strings separated by commas:
def rdd_line_to_string(x):
#    (i, a, s, c) = x # Unpack the tuple
    (i, a, s, c) = x # Unpack the tuple
#    return str(i) + ',' + a + ',' + s + ',' + str(c)
    return str(i) + '|' + str(a) + '|' + str(s) + '|' + str(c)


def resilient_json(s):
    try:
        return json.loads(s)
    except:
        return {}


def save_data_from_subreddit_month_year_tsv(type_txt, month, year, subreddit):
    if type_txt == 'submissions':
        type_data = 'RS'
    else:
        type_data = 'RC'
    if year == '2010':
        data_path = f'/data/shared/reddit/{type_txt}/{year}/{type_data}_*{year}-{month}.*'
    else:
        data_path = f'/data/shared/reddit/{type_txt}/{year}/{type_data}_*{year}-{month}.bz2'


    data = (
        # All of the following objects are pyspark.RDD, a (distributed) unordered set of objects (call them "lines").
        # `sc.textFile(data_path)` reads the data from the path on the server and store them on a SparkContext object.
        sc.textFile(data_path) # The read data are now stored in a RDD, where each line is a string.
        .map(resilient_json) # Apply the function resilient_json to each row: each line become a dictionary.
        .map(lambda x: (x.get("author"),
                x.get("subreddit"),
                x.get("body"),
                x.get("selftext"),
                x.get("created_utc"),
                x.get("id"),
                x.get("link_id"),
                x.get("score"),
                x.get("subreddit_id"),
                x.get("parent_id")))
        .filter(lambda x: x[2] == subreddit) # This filters out x for which subreddit is not PoliticalCompass.
    )

    output_path_temp = '../data/TempUsers'
    if os.path.isdir(output_path_temp):
        shutil.rmtree(output_path_temp)
    output_path_tsv = f'../data/strong_activation_{type_txt}/{subreddit}_{year}_{month}_{type_data}.tsv'

    data.map(rdd_line_to_string).saveAsTextFile(output_path_temp)
    
    with open(output_path_tsv, 'w') as outfile:
        for infilename in glob(output_path_temp + "/part-*"):
            with open(infilename) as infile:
                shutil.copyfileobj(infile, outfile)
    
    shutil.rmtree(output_path_temp)





if __name__ == '__main__':
    type_txt, month, year, subreddit = sys.argv[1:]
    
    print(subreddit, year, month)
    save_data_from_subreddit_month_year_tsv(type_txt, month, year, subreddit)
