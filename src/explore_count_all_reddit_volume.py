
import sys
sys.path += ["../src"]
import spark_init

sc = spark_init.spark_context()

data_path = "/data/big/xxx/climact/data/"


if __name__ == '__main__':
    month_years = [f"{year}_{month:02d}" for year in range(2012, 2024) for month in range(1,13)]
    for month_year in month_years:
        for type_text in ["comments", "submissions"]:
            for type_txt in ["RC", "RS"]:
                path = f'/data/shared/reddit/{type_text}/{month_year.split("_")[0]}/{type_txt}_{month_year.split("_")[0]}-{month_year.split("_")[1]}.bz2'
                count_contents = sc.textFile(path).count()
                with open(data_path + "reddit_volume.txt", "a") as f:
                    f.write(" ".join([month_year, type_txt, str(count_contents), "\n"]))
