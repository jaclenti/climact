import pandas as pd
import sys
from datetime import datetime
from glob import glob
import warnings
from tqdm import tqdm
import json
warnings.simplefilter(action='ignore', category=FutureWarning)


sys.path += ['../src/']

climate_themes = ["mitigation", "environment", "climate", "env_", "natural_disaster"]
data_path = "/data/big/xxx/climact/data/"
gdelt_path = "/data/shared/xxx/projects/test/data/"

column_names = ['GKGRECORDID', 'DATE', 'SourceCollectionIdentifier', 'SourceCommonName',
       'DocumentIdentifier', 'Counts', 'V2Counts', 'Themes', 'V2Themes',
       'Locations', 'V2Locations', 'Persons', 'V2Persons', 'Organizations',
       'V2Organizations', 'V2Tone', 'Dates', 'GCAM', 'SharingImage',
       'RelatedImages', 'SocialImageEmbeds', 'SocialVideoEmbeds', 'Quotations',
       'AllNames', 'Amounts', 'TranslationInfo', 'Extras']

if __name__ == '__main__':
    bad_dates = []
    with open("/data/big/xxx/climact/data/bad_files_gdelt.txt", "r") as f:
        for line in f.readlines():
            bad_dates.append(json.loads(line))
    bad_files = []
    all_corrupted_files = []
    count_corrupted_files = []
    for date in bad_dates:
        corrupted_files, good_files = 0, 0
        day, month, year = date
        files = sorted(glob(f"{gdelt_path}{year}{month:02d}{day:02d}*zip"))
        date_df = []
        if len(files) > 0:
            for file in tqdm(files):
                try:
                    # date_df.append(pd.read_csv(file, compression = "zip", sep='\t',
                    # header = None, index_col = 0, names = column_names).dropna(subset = "V2Themes")\
                    #     .assign(is_climate = lambda x: [any(s in u.lower() for s in climate_themes) for u in x["V2Themes"]]).query("is_climate")\
                    #         [['DATE', 'SourceCommonName','DocumentIdentifier', 'Themes', 'V2Themes','V2Locations']])
                    pd.read_csv(file, compression = "zip", sep='\t', header = None, index_col = 0, names = column_names)
                    good_files += 1
                    
                except:
                    corrupted_files += 1
                    all_corrupted_files.append(file)
            
            # if good_files > 0:
            #     pd.concat(date_df).to_csv(f"{data_path}gdelt_climate_themes/{year}_{month:02d}_{day:02d}.csv.zip",  compression = "zip")
        bad_files.append(date + [good_files, corrupted_files])
        print(date + [good_files, corrupted_files])
    pd.DataFrame(bad_files, columns = ["day", "month", "year", "good_file", "bad_files"]).to_csv(f"{data_path}count_bad_files.csv")  

    with open(f"{data_path}all_corrupted_files_gdelt.txt", "w") as f:
        for file in all_corrupted_files:
            f.write(file + "\n")
                

