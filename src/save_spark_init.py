# Configure the necessary Spark environment
import os
import sys

# Path for java
os.environ['JAVA_HOME'] = "/usr/lib/jvm/java-11-openjdk-amd64"

spark_home = "/opt/spark/"
sys.path.insert(0, spark_home + "/python")

# Add the py4j to the path.
sys.path.insert(0, spark_home + "python/lib/py4j-0.10.9.3-src.zip")

# Path for spark source folder
os.environ['SPARK_HOME'] = spark_home

from pyspark import SparkContext, SparkConf, SQLContext

conf = SparkConf()


conf.setMaster("local[10]").setAppName("JLenti Spark App")
conf.set('spark.executor.cores', '16')
#conf.setMaster("spark://igea:7077").setAppName("Test app")
conf.set("spark.executor.memory", "8g")
conf.set("spark.driver.memory", "12g")
conf.set("spark.ui.port", "4867")
conf.set('spark.driver.maxResultSize','24g')
conf.set("spark.local.dir", "/data/fast/tmp/")

def spark_context():
     return SparkContext(conf=conf)
    
def sql_context(sc):
     return SQLContext(sc)
