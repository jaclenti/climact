import pandas as pd
import sys
from datetime import datetime
from glob import glob
from gdelt import gdelt
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


sys.path += ['../src/']

climate_themes = ["mitigation", "environment", "climate", "env_", "natural_disaster"]
data_path = "/data/shared/xxx/climact/data/"
gdelt_path = "/data/shared/xxx/projects/test/data/"

column_names = ['GKGRECORDID', 'DATE', 'SourceCollectionIdentifier', 'SourceCommonName',
       'DocumentIdentifier', 'Counts', 'V2Counts', 'Themes', 'V2Themes',
       'Locations', 'V2Locations', 'Persons', 'V2Persons', 'Organizations',
       'V2Organizations', 'V2Tone', 'Dates', 'GCAM', 'SharingImage',
       'RelatedImages', 'SocialImageEmbeds', 'SocialVideoEmbeds', 'Quotations',
       'AllNames', 'Amounts', 'TranslationInfo', 'Extras']


def save_gdelt_news(day, month, year, files_day):
    if len(files_day)>0:
        news = pd.concat([pd.read_csv(file, compression = "zip", sep='\t', encoding="latin-1",
                    header = None, index_col = 0, names = column_names).dropna(subset = "V2Themes")\
                        .assign(is_climate = lambda x: [any(s in u.lower() for s in climate_themes) for u in x["V2Themes"]]).query("is_climate")\
                        [['DATE', 'SourceCommonName',
                        'DocumentIdentifier', 'Themes', 
                        'V2Themes','V2Locations']] for file in files_day])
        print(day, month, year, len(news))
        news.to_csv(f"{data_path}gdelt_climate_themes/{year}_{month:02d}_{day:02d}.csv.zip",  compression = "zip")

if __name__ == '__main__':
    version = sys.argv[1]
    if version == "2":
        files = sorted(glob(f"{gdelt_path}*zip"))
        bad_files = []
        for year in range(2023,2025):
            for month in range(1, 13):
                files_month = [file for file in files if f"/{year}{month:02d}" in file]
                for day in range(1,32):
                    files_day = [file for file in files if f"/{year}{month:02d}{day:02d}" in file]
                    
                    try:
                        save_gdelt_news(day, month, year, files_day)
                    except:
                        print("error")
                        bad_files.append([day, month, year])
        with open(f"{data_path}bad_files_gdelt.txt", "w") as f:
            for s in bad_files:
                f.write(str(s) +"\n")
    if version == "1":
        gd1 = gdelt(version = 1)
        for date in pd.date_range('2013-04-01', '2015-03-15').strftime('%d-%m-%y'):
            print(date)
            day, month, year = date.split("-")
            news = gd1.Search([f"{month}-{day}-{year}"], table = "gkg", coverage = True, output = "df", normcols=True).dropna(subset = "themes")\
                .assign(is_climate = lambda x: [any(s in u.lower() for s in climate_themes) for u in x["themes"]], id = None, v2themes = None).query("is_climate")\
                [["date", "sources", "id", "themes", "v2themes", "locations"]]
            print(day, month, year, len(news))
                        # filter = [("environment" in u.lower() or "env_" in u.lower() or "climate" in u.lower() or "natural_disaster" in u.lower() or "mitigation" in u.lower()) for u in list(news.dropna(subset = "themes")["themes"])]
                        # news = news.query("@filter").assign(id = None, v2themes = None)[["date", "sources", "id", "themes", "v2themes", "locations"]]
            news.to_csv(f"{data_path}gdelt_climate_themes_v1/{year}_{month}_{day}.csv.zip",  compression = "zip")

                    
                    
                
            

                    

