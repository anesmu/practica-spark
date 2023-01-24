from pyspark.sql import SparkSession
import util
from pyspark.sql.functions import *
from pyspark.sql.types import StringType


def main(spark):
    path = 'data/'
    filename_google = 'GoogleProducts.csv'
    filename_amazon = 'Amazon.csv'

    absolute_file_path_google = util.get_absolute_file_path(path, filename_google)
    absolute_file_path_amazon = util.get_absolute_file_path(path, filename_amazon)

    df_google = spark.read.csv(absolute_file_path_google, header=True, inferSchema=True)
    df_amazon = spark.read.csv(absolute_file_path_amazon, header=True, inferSchema=True)
    df_products = df_google.union(df_amazon)
    df_products.show(1000)
    df_products = df_products.select([coalesce(col, lit(None)).alias(col) for col in df_products.columns])
    df_products.show(1000)
    df_products = df_products.select([lower(col).alias(col) for col in df_products.columns])
    df_products.show(1000)
    df_products = df_products.select(
        [regexp_replace(col, "[^a-zA-Z0-9]", "").alias(col) for col in df_products.columns if
         df_products.schema[col].dataType == StringType()])
    df_products.show(1000)


if __name__ == '__main__':
    spark = SparkSession.builder.appName("Offers ") \
        .master("local[*]").getOrCreate()

    spark.sparkContext.setLogLevel('warn')
    main(spark)
    spark.stop()
