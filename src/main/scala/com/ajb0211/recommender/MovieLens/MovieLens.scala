package com.ajb0211.recommender.MovieLens

import scala.util.Random
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.recommendation.{ALS, ALSModel}
import org.apache.spark.sql.{DataFrame, Dataset, SparkSession}
import org.apache.spark.sql.types._

object MovieLens extends App{
  // If run locally, similar to Python:
  //   if __name__ == "__main__"

  val spark: SparkSession = SparkSession.
    builder.
    master("local[*]").
    appName("MovieLens").
    getOrCreate

  spark.sparkContext.setLogLevel("WARN")

  val dataDir: String = "./src/main/resources/ml-100k/"

  val ml: MovieLens = new MovieLens(spark,
    dataDir + "u.data",
    dataDir + "u.user",
    dataDir + "u.item")


  ml.ratingData.show

}

class MovieLens(
         val spark: SparkSession,
         val ratingFile: String,
         val itemFile: String,
         val userFile: String,
         val partitions: Int = 4
                             ) {
  /*
    Base class that standardizes usage of MovieLens data and files
    Structure allows usage for both local data set and AWS EC2 instances
    Can be extended or traited for portability of labeling scheme
  */



  // Required to use methods like toDF
  import spark.implicits._

  // Lazy evaluation, will not evaluate if child class does not use these objects
  lazy val ratingData: DataFrame = parseRatings(ratingFile)
  lazy val itemData: DataFrame = parseItems(itemFile)
  lazy val userData: DataFrame = parseUsers(userFile)


  // Read the rating file, such as u.data
  // Testing in ratings.spark
  def parseRatings(fileString: String): DataFrame = {
    spark.sparkContext.textFile(fileString)
      .map{ line =>
        val Array(user,item,rating,timestamp) = line.split('\t')
        (user.toInt,item.toInt,rating.toInt,timestamp.toInt)
      }.toDF("userID","itemID","rating","unixTime")
  }

  // Read the movie information file, such as u.item
  // Testing in items.spark
  def parseItems(fileString: String): DataFrame = {
    /*
    explicit StructType declaration must be used here
    scala limits tuple and unpacking to 22 items
    this data set is 23 columns wide

    Genres are stored as label encoded 1/0 in a wide format
     */


    val itemSchema = StructType(
        StructField("itemID",            IntegerType, nullable = false) ::
        StructField("title",             StringType,  nullable = false) ::
        StructField("releaseDate",       StringType,  nullable = false) ::
        StructField("videoReleaseDate",  StringType,  nullable = true)  ::
        StructField("imdb",              StringType,  nullable = true)  ::
        StructField("action",            IntegerType, nullable = true)  ::
        StructField("adventure",         IntegerType, nullable = true)  ::
        StructField("animation",         IntegerType, nullable = true)  ::
        StructField("children",          IntegerType, nullable = true)  ::
        StructField("comedy",            IntegerType, nullable = true)  ::
        StructField("crime",             IntegerType, nullable = true)  ::
        StructField("documentary",       IntegerType, nullable = true)  ::
        StructField("drama",             IntegerType, nullable = true)  ::
        StructField("fantasy",           IntegerType, nullable = true)  ::
        StructField("noir",              IntegerType, nullable = true)  ::
        StructField("horror",            IntegerType, nullable = true)  ::
        StructField("musical",           IntegerType, nullable = true)  ::
        StructField("mystery",           IntegerType, nullable = true)  ::
        StructField("romance",           IntegerType, nullable = true)  ::
        StructField("scifi",             IntegerType, nullable = true)  ::
        StructField("thriller",          IntegerType, nullable = true)  ::
        StructField("war",               IntegerType, nullable = true)  ::
        StructField("western",           IntegerType, nullable = true)  ::
        Nil
    )

    spark.read.
      option("delimiter", "|").
      schema(itemSchema).
      csv(fileString)
  }

  // Read the user data file, such as u.user
  // Testing in users.spark
  def parseUsers(fileString: String): DataFrame = {
    spark.sparkContext.textFile(fileString).
      map{ line =>
        val Array(user,age,gender,occupation,zipcode) = line.split('|')
        (user.toInt,age.toInt,gender,occupation,zipcode)
      }.toDF("userID","age","sex","occupation","zipcode")
  }

  def initializeModel(
                       rank: Int,
                       regParam: Double,
                       alpha: Double,
                       maxIter: Int
                     ): ALS = new ALS().
    setSeed(Random.nextLong()).
    setImplicitPrefs(true).
    setRank(rank).
    setRegParam(regParam).
    setAlpha(alpha).
    setMaxIter(maxIter).
    setUserCol("userID").
    setItemCol("itemID").
    setRatingCol("rating").
    setPredictionCol("prediction")


}
