import pandas as pd
import sys
sys.path += ["../src"]
import climact_shared.src.utils as cu
from glob import glob
import re


daily_news = pd.read_csv(cu.data_path + "count_all_gdelt_news.csv", index_col = 0)
daily_news["date"] = pd.to_datetime(daily_news["date"])
daily_news = daily_news.assign(str_date = lambda x: [u.strftime("%Y_%m_%d") for u in x["date"]])
def get_us_states_and_ca_region(line):
    pattern = "(US\w\w|CA\d\d)$"
    return [(split_u[:2], split_u[2:]) for u in line.split(";") for split_u in u.split("#") if re.match(pattern = pattern, string = split_u)] if line == line else []


def get_data_strike_natural_disaster_admin(date):
    # date in format 2017_01_01
    df = pd.read_csv(sorted(glob(cu.data_path.replace("big", "shared") + f"gdelt_climate_themes/{date}*"))[0])\
        .assign(admin1code = lambda x: [list(set(get_us_states_and_ca_region(u))) for u in x["V2Locations"]],
                is_strike = lambda x: ["STRIKE" in u.replace(",", ";").split(";") for u in x["V2Themes"]], # want exact match, theme STRIKE
                is_climate_action = lambda x: ["UNGP_CLIMATE_CHANGE_ACTION" in u for u in x["V2Themes"]], 
                is_climate = lambda x: ["CLIMATE" in u for u in x["V2Themes"]],
                is_natural_disaster = lambda x: ["NATURAL_DISASTER" in u for u in x["V2Themes"]])[["admin1code","is_strike","is_climate_action","is_climate","is_natural_disaster"]]\
                    .explode("admin1code").dropna().value_counts().reset_index()\
                        .assign(country = lambda x: [u[0] for u in x["admin1code"]],
                                region = lambda x: [u[1] for u in x["admin1code"]],
                                norm_count = lambda x: x["count"] / daily_news.set_index("str_date").loc[date,"count"])
    return df


if __name__ == '__main__':
    for day in daily_news["str_date"]:
        print(day)
        count_news = get_data_strike_natural_disaster_admin(day)
        count_news.to_csv(cu.data_path + f"us_canada_news_daily_count/{day}.csv")



