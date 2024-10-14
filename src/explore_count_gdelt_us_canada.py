import pandas as pd
import sys
sys.path += ["../src"]
from datetime import datetime
from glob import glob
# from gdelt import gdelt
import climact_shared.src.utils as cu
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import re

sys.path += ['../src/']

gdelt_path = "/data/shared/xxx/projects/test/data/"

column_names = ['GKGRECORDID', 'DATE', 'SourceCollectionIdentifier', 'SourceCommonName',
       'DocumentIdentifier', 'Counts', 'V2Counts', 'Themes', 'V2Themes',
       'Locations', 'V2Locations', 'Persons', 'V2Persons', 'Organizations',
       'V2Organizations', 'V2Tone', 'Dates', 'GCAM', 'SharingImage',
       'RelatedImages', 'SocialImageEmbeds', 'SocialVideoEmbeds', 'Quotations',
       'AllNames', 'Amounts', 'TranslationInfo', 'Extras']

def get_us_states_and_ca_region(line):
    pattern = "(US\w\w|CA\d\d)$"
    return [(split_u[:2], split_u[2:]) for u in line.split(";") for split_u in u.split("#") if re.match(pattern = pattern, string = split_u)] if line == line else []


def count_news_us_canada(file):
    try: 
        df = (pd.read_csv(file, compression = "zip", sep='\t',
                           header = None, index_col = 0, names = column_names)
                           .dropna(subset = "V2Themes")
                           [['DATE', 'SourceCommonName',
                             'DocumentIdentifier', 'Themes', 
                             'V2Themes','V2Locations']]
                             .assign(country_admin = lambda x: [list(set(get_us_states_and_ca_region(u))) for u in x["V2Locations"]])
                             .explode("country_admin")
                             .dropna()
                             .assign(country = lambda x: [u[0] for u in x["country_admin"]], 
                                     admin1code = lambda x: [u[1] for u in x["country_admin"]])
                                     [["country", "admin1code"]]
                             .value_counts()
                             .reset_index()
                    )
        return df 
    except:
        print("bad file")
        return pd.DataFrame()            
    
    



# def save_gdelt_news(day, month, year, files_day):
#     if len(files_day)>0:
#         news = pd.concat([pd.read_csv(file, compression = "zip", sep='\t',
#                     header = None, index_col = 0, names = column_names).dropna(subset = "V2Themes")\
#                         .assign(is_climate = lambda x: [any(s in u.lower() for s in climate_themes) for u in x["V2Themes"]]).query("is_climate")\
#                         [['DATE', 'SourceCommonName',
#                         'DocumentIdentifier', 'Themes', 
#                         'V2Themes','V2Locations']] for file in files_day])
#         print(day, month, year, len(news))
#         news.to_csv(f"{data_path}gdelt_climate_themes/{year}_{month:02d}_{day:02d}.csv.zip",  compression = "zip")

if __name__ == '__main__':
    files = sorted(glob(f"{gdelt_path}*zip"))
    # print(len(files))
    for year in range(2015, 2023):
        for month in range(1, 13):
            # print(year, month)
            files_month = [file for file in files if f"/{year}{month:02d}" in file]
            print(len(files_month))
            for day in range(1,32):
                files_day = [file for file in files if f"/{year}{month:02d}{day:02d}" in file]
                print(year, month, day, len(files_day))

                news_day = [count_news_us_canada(file) for file in files_day]
                if len(news_day) > 0:
                    (pd.concat(news_day).groupby(["country", "admin1code"]).sum()
                     .reset_index()
                     .to_csv(cu.data_path + f"news_count_us_canada/{year}{month:02d}{day:02d}.csv.gz", compression = "gzip"))
                else:
                    print("Null")
