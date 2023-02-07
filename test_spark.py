import os

import Candidate_score
import Characterization
import Constants
import Entity_gen
import Standardization
import Util

from pyspark.sql import SparkSession


def main(spark):
    # Part 0: Load df from  /data
    df = Util.load_df_from_data_folder(spark)

    # Part 1: standardization
    df_standard = Standardization.standardization_df(df)

    # Part 2: characterization
    df_characterized = Characterization.characterization_idf(df_standard)
    df_standard.show()
    # Part 3: Candidate_pairs
    df_candidates = Entity_gen.entity_gen(df_characterized)

    # Part 4: Get map coincidences
    df_map = Candidate_score.get_pair_neighbor(df_characterized, df_candidates)

    # Part 5: Save results
    path = Util.get_absolute_file_path(Constants.path, Constants.result_path)
    df_map.write.csv(path, mode="overwrite", header=True)

    return


if __name__ == '__main__':
    os.environ["PYSPARK_SUBMIT_ARGS"] = "--packages graphframes:graphframes:0.8.1-spark3.0-s_2.12 pyspark-shell"
    spark_session = SparkSession.builder.appName("Offers ") \
        .master("local[*]") \
        .getOrCreate()
    spark_session._sc.setCheckpointDir("checkpoints")
    spark_session.sparkContext.setLogLevel('warn')
    main(spark_session)
    spark_session.stop()
