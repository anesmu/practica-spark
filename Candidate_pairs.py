import numpy as np
from pyspark.sql.functions import udf, col
from pyspark.sql.types import DoubleType
from scipy.spatial.distance import cosine


def cosine_similarity(v1, v2):
    v1 = np.array(v1.toArray())
    v2 = np.array(v2.toArray())
    cos_dis = float(cosine(v1, v2))
    return round((1 - cos_dis) * 100, 2)


def get_distance_df(df):
    cos_sim = udf(cosine_similarity, DoubleType())
    df_amazon = df.filter(col("id").like("%http%"))
    df_google = df.filter(~(col("id").like("%http%")))
    df_distance = df_amazon.alias("df1").crossJoin(df_google.alias("df2")) \
        .where(col("df1.id") != col("df2.id")) \
        .select(col("df1.id").alias("id1"), col("df2.id").alias("id2"),
                cos_sim(col("df1.name_vector"), col("df2.name_vector")).alias("name_distance"),
                cos_sim(col("df1.description_vector"), col("df2.description_vector")).alias("description_distance"),
                cos_sim(col("df1.manufacturer_vector"), col("df2.manufacturer_vector")).alias(
                    "manufacturer_distance"),
                cos_sim(col("df1.price_vector"), col("df2.price_vector")).alias("price_distance"))
    return df_distance


def calculate_mean_cosine(df):
    df = df.withColumn("mean", (df["name_distance"] + df["description_distance"] + df["manufacturer_distance"] +
                                df["price_distance"]) / 4)
    df = df.filter(df["mean"] > 10)

    return df


def get_pair_cadidates(df):
    df_distance = get_distance_df(df)
    df_distance_means = calculate_mean_cosine(df_distance)

    return df_distance_means
