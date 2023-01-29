import os

import Constants


def load_df_from_data_folder(spark):
    absolute_file_path_google = get_absolute_file_path(Constants.path, Constants.filename_google)
    absolute_file_path_amazon = get_absolute_file_path(Constants.path, Constants.filename_amazon)

    df_google = spark.read.csv(absolute_file_path_google, header=True, inferSchema=True)
    df_amazon = spark.read.csv(absolute_file_path_amazon, header=True, inferSchema=True)
    df = union_df(df_amazon, df_google)

    return df


# All bellow are private methods that only are use on first method

def union_df(df_amazon, df_google):
    df_google = df_google.withColumnRenamed("title", "name")
    df = df_google.union(df_amazon)

    return df


def get_absolute_file_path(path, filename):
    current_dir = os.path.dirname(__file__)
    relative_path = "{}{}".format(path, filename)
    absolute_file_path = os.path.join(current_dir, relative_path)
    return absolute_file_path
