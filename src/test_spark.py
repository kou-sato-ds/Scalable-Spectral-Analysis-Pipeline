from pyspark.sql import SparkSession
spark = SparkSession.builder.master("spark://spark-master:7077").getOrCreate()
print("🎉 Spark Session Created!")
spark.stop()