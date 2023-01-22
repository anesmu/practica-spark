from pyspark.sql import SparkSession
import util
import pandas as pd


def main(spark):
    path = 'data/'
    filename_google = 'GoogleProducts.csv'
    filename_amazon = 'Amazon.csv'

    absolute_file_path_google = util.get_absolute_file_path(path, filename_google)
    absolute_file_path_amazon = util.get_absolute_file_path(path, filename_amazon)

    df_google = spark.read.csv(absolute_file_path_google, header=True, inferSchema=True)
    df_amazon = spark.read.csv(absolute_file_path_amazon, header=True, inferSchema=True)
    df_products = df_google.union(df_amazon)
    df_products.write.csv("/Users/anesmu/Desktop/spark/data/output.csv", header=True, mode="overwrite", sep=",")
    #df_products.show(5)
    #df_products.printSchema()


if __name__ == '__main__':
    spark = SparkSession.builder.appName("Offers ") \
        .master("local[*]").getOrCreate()

    spark.sparkContext.setLogLevel('warn')
    main(spark)
    spark.stop()
