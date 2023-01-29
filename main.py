import Candidate_pairs
import Characterization
import Standardization
import Util

from pyspark.sql import SparkSession


def main(spark):
    # Load df from  /data
    df = Util.load_df_from_data_folder(spark)

    # Part 1: standardization
    df = Standardization.standardization_df(df)

    # Part 2: characterization
    df = Characterization.characterization_idf(df)

    # Part 3: Candidate_pairs
    df_distance = Candidate_pairs.get_pair_cadidates(df)
    df_distance.write.csv("/Users/anesmu/Desktop/spark/data/df_distance", header=True, mode="overwrite")

    df_distance.show(100, truncate=False)

    return


if __name__ == '__main__':
    spark_session = SparkSession.builder.appName("Offers ") \
        .master("local[*]") \
        .getOrCreate()

    spark_session.sparkContext.setLogLevel('warn')
    main(spark_session)
    spark_session.stop()
