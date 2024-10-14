import pandas as pd
import numpy as np
from glob import glob
import sys
sys.path += ["../src"]
import climact_shared.src.utils as cu
import os
import spark_init
import pyspark.sql.functions as F
from pyspark.sql import SQLContext
import os

sc = spark_init.spark_context()
sqlContext = SQLContext(sc)

if __name__ == "__main__":
    for file in sorted(glob(cu.data_path + "histories_after_strong/RS*")):
        print(file.split("/")[-1].split(".")[-2])
        # np.save(cu.data_path + f"parent_id_after_strong/{file.replace('.', '/').split('/')[-2]}.npy", pd.read_parquet(file).parent_id.unique())
        if file.split("/")[-1].split("_")[0] == "RS":
            for item in ["id"]: #, "parent_id"]:
                path_name = cu.data_path + f"{item}_after_strong/{file.replace('.', '/').split('/')[-2]}.npy"
                if not os.path.exists(path_name):
                    if os.stat(file).st_size != 4096:
                        np.save(path_name, sqlContext.read.parquet(file).select(item).distinct().toPandas()[item])
    # for file in sorted(glob(cu.data_path + "histories_after_strong/RC*")):
    #     print(file.split("/")[-1].split(".")[-2])
    #     # np.save(cu.data_path + f"parent_id_after_strong/{file.replace('.', '/').split('/')[-2]}.npy", pd.read_parquet(file).parent_id.unique())
    #     if file.split("/")[-1].split("_")[0] == "RC":
    #         for item in ["id", "link_id"]: #, "parent_id"]:
    #             path_name = cu.data_path + f"{item}_after_strong/{file.replace('.', '/').split('/')[-2]}.npy"
    #             if not os.path.exists(path_name):
    #                 if os.stat(file).st_size != 4096:
    #                     np.save(path_name, sqlContext.read.parquet(file).select(item).distinct().toPandas()[item])
# if __name__ == "__main__":
#     for file in sorted(glob(cu.data_path + "histories_after_strong/RC*")):
#         print(file.split("/")[-1].split(".")[-2])
#         # np.save(cu.data_path + f"parent_id_after_strong/{file.replace('.', '/').split('/')[-2]}.npy", pd.read_parquet(file).parent_id.unique())
#         if file.split("/")[-1].split("_")[0] == "RC":
#             path_name = cu.data_path + f"parent_id_after_strong/{file.replace('.', '/').split('/')[-2]}.npy"
#             if not os.path.exists(path_name):
#                 if os.stat(file).st_size != 4096:
#                     np.save(path_name, sqlContext.read.parquet(file).select("parent_id").distinct().toPandas()["parent_id"])
        