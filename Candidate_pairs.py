import numpy as np
from pyspark.sql.functions import col, udf
from pyspark.sql.types import DoubleType
from scipy.spatial.distance import cosine


def cosine_similarity(v1, v2):
    v1 = np.array(v1.toArray())
    v2 = np.array(v2.toArray())
    return float(cosine(v1, v2))


def get_distance_df(df):
    cos_sim = udf(cosine_similarity, DoubleType())

    df_distance = df.alias("df1").crossJoin(df.alias("df2")) \
        .where(col("df1.id") != col("df2.id")) \
        .select(col("df1.id").alias("id1"), col("df2.id").alias("id2"),
                cos_sim(col("df1.name_vector"), col("df2.name_vector")).alias("name_distance"),
                cos_sim(col("df1.description_vector"), col("df2.description_vector")).alias("description_distance"),
                cos_sim(col("df1.manufacturer_vector"), col("df2.manufacturer_vector")).alias(
                    "manufacturer_distance"),
                cos_sim(col("df1.price_vector"), col("df2.price_vector")).alias("price_distance"))

    return df_distance
