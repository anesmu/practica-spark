import numpy as np
from pyspark.sql.functions import udf, col
from pyspark.sql.types import DoubleType
from scipy.spatial.distance import cosine
from pyspark.sql.functions import round as r


def get_pair_neighbor(df, df_candidates):
    df_mixed = gen_df(df, df_candidates)
    df_distance = get_distance_df(df_mixed)
    df_distance_means = calculate_mean_cosine(df_distance)
    df_map = df_distance_means.select("idGoogle", "idAmazon")

    return df_map


# All bellow are private methods that only are use on first method


def cosine_similarity(v1, v2):
    v1 = np.array(v1.toArray())
    v2 = np.array(v2.toArray())
    cos_dis = float(cosine(v1, v2))
    return round((1 - cos_dis) * 100, 2)


def get_distance_df(df):
    cos_sim = udf(cosine_similarity, DoubleType())

    df_distance = df.select(col("idGoogle"), col("idAmazon"),
                            cos_sim(col("nameGoogle"), col("nameAmazon")).alias("name_distance"),
                            cos_sim(col("descriptionGoogle"), col("descriptionAmazon")).alias(
                                "description_distance"),
                            cos_sim(col("manufacturerGoogle"), col("manufacturerAmazon")).alias(
                                "manufacturer_distance"),
                            cos_sim(col("priceGoogle"), col("priceAmazon")).alias("price_distance"))
    return df_distance


def calculate_mean_cosine(df):
    df = df.withColumn("mean", r(((df["name_distance"] + df["description_distance"] + df["manufacturer_distance"] +
                                   df["price_distance"]) / 4), 2))
    df = df.filter(df["mean"] > 37.5)
    return df


def gen_df(df, df_candidates):
    df_result = df_candidates.join(df, df_candidates.idGoogle == df.id).select(df_candidates.idGoogle,
                                                                               df_candidates.idAmazon,
                                                                               df.name_vector.alias("nameGoogle"),
                                                                               df.description_vector.alias(
                                                                                   "descriptionGoogle"),
                                                                               df.manufacturer_vector.alias(
                                                                                   "manufacturerGoogle"),
                                                                               df.price_vector.alias("priceGoogle"))
    df_result = df_result.join(df, df_result.idAmazon == df.id).select(df_result.idGoogle, df_result.idAmazon,
                                                                       df_result.nameGoogle,
                                                                       df_result.descriptionGoogle,
                                                                       df_result.manufacturerGoogle,
                                                                       df_result.priceGoogle,
                                                                       df.name_vector.alias("nameAmazon"),
                                                                       df.description_vector.alias("descriptionAmazon"),
                                                                       df.manufacturer_vector.alias(
                                                                           "manufacturerAmazon"),
                                                                       df.price_vector.alias("priceAmazon"))
    return df_result
