import pandas as pd
import sys
from datetime import datetime
from glob import glob
# from gdelt import gdelt
import os
import warnings
import subprocess
import json
warnings.simplefilter(action='ignore', category=FutureWarning)


sys.path += ['../src/']

data_path = "/data/big/xxx/climact/data/"
gdelt_path = "/data/shared/xxx/projects/test/data/"


def count_rows(file):
    try:
        command = f"zcat {file}|wc -l"
        rows = json.loads(subprocess.run(command, shell=True, capture_output=True, text=True).stdout)
        
        
    except:
        rows = 0
        with open(f"{data_path}bad_files_all_gdelt_count_rows.txt", "a") as f:
            f.write(file+"\n")
                
    
    return rows



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
    count_rows_all = []
    for year in range(2015, 2024):
        for month in range(1, 13):
            print(year, month)
            files_month = [file for file in files if f"/{year}{month:02d}" in file]
            for day in range(1,32):
                count_rows_day = 0
                files_day = [file for file in files if f"/{year}{month:02d}{day:02d}" in file]
                for file in files_day:
                    rows = count_rows(file)
                    count_rows_day += rows
                count_rows_all.append([count_rows_day, f"{year}{month:02d}{day:02d}"])

                with open(f"{data_path}count_all_gdelt_news.txt", "a") as f:
                    f.write(f"{count_rows_day} {year}{month:02d}{day:02d}" +"\n")
                
                    
    #     with open(f"{data_path}count_all_gdelt_news.txt", "w") as f:
    #         for s in count_rows_all:
    #             f.write(str(s) +"\n")
    # if version == "1":
    #     gd1 = gdelt(version = 1)
    #     for date in pd.date_range('2013-04-01', '2015-03-15').strftime('%d-%m-%y'):
    #         print(date)
    #         day, month, year = date.split("-")
    #         news = gd1.Search([f"{month}-{day}-{year}"], table = "gkg", coverage = True, output = "df", normcols=True).dropna(subset = "themes")\
    #             .assign(is_climate = lambda x: [any(s in u.lower() for s in climate_themes) for u in x["themes"]], id = None, v2themes = None).query("is_climate")\
    #             [["date", "sources", "id", "themes", "v2themes", "locations"]]
    #         print(day, month, year, len(news))
    #                     # filter = [("environment" in u.lower() or "env_" in u.lower() or "climate" in u.lower() or "natural_disaster" in u.lower() or "mitigation" in u.lower()) for u in list(news.dropna(subset = "themes")["themes"])]
    #                     # news = news.query("@filter").assign(id = None, v2themes = None)[["date", "sources", "id", "themes", "v2themes", "locations"]]
    #         news.to_csv(f"{data_path}gdelt_climate_themes_v1/{year}_{month}_{day}.csv.zip",  compression = "zip")

    #     # for year in range(2013, 2016):
    #     #     for month in range(1, 13):
    #     #         for day in range(1,32):
    #     #             try:
    #                 #     news = gd1.Search([f"{day:02d}-{month:02d}-{year}"], table = "gkg", coverage = True, output = "df", normcols=True).dropna(subset = "themes")\
    #                 #     .assign(is_climate = lambda x: [any(s in u.lower() for s in climate_themes) for u in x["themes"]], id = None, v2themes = None).query("is_climate")\
    #                 #     [["date", "sources", "id", "themes", "v2themes", "locations"]]
    #                 #     print(day, month, year, len(news))
    #                 #     # filter = [("environment" in u.lower() or "env_" in u.lower() or "climate" in u.lower() or "natural_disaster" in u.lower() or "mitigation" in u.lower()) for u in list(news.dropna(subset = "themes")["themes"])]
    #                 #     # news = news.query("@filter").assign(id = None, v2themes = None)[["date", "sources", "id", "themes", "v2themes", "locations"]]
    #                 #     news.to_csv(f"{data_path}gdelt_climate_themes_v1/{year}_{month}_{day}.csv.zip",  compression = "zip")
    #                 # except:
    #                 #     print(day, month, year, "error")
                    
                    
                
            

                    

