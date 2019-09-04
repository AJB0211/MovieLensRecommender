package com.ajb0211.recommender

import org.apache.spark.broadcast.Broadcast

import scala.util.Random
import org.apache.spark.sql.{DataFrame, Dataset, SparkSession}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._

import scala.collection.mutable.ArrayBuffer

// Future work: more metrics with a standardized interface
// (maybe as classes with an inherited base class?)
// for applying different metrics to the model evaluation scheme in ALSTrainer.scala
package object metrics {


  def areaUnderCurve( X: String,
                      y: String,
                      testData: DataFrame,
                      B_allItems: Broadcast[Array[Int]],
                      // generally this will be model.transform
                      predictFunction: (DataFrame => DataFrame)): Double = {
    /*
      Modified from Advanced Analytics with Spark 2/E
      O'Reilly
      Chapter 3

      Modifications largely made to generalize function and move it to an external package
      rather than make it a class method
    */

    // leverage SparkSession object stored with testData DataFrame to import implicits
    import testData.sparkSession.implicits._

    // What this actually computes is AUC, per user. The result is actually something
    // that might be called "mean AUC".

    // Take held-out data as the "positive".
    // Make predictions for each of them, including a numeric score
    val positivePredictions = predictFunction(testData.select(X, y)).
      withColumnRenamed("prediction", "positivePrediction")

    // BinaryClassificationMetrics.areaUnderROC is not used here since there are really lots of
    // small AUC problems, and it would be inefficient, when a direct computation is available.

    // Create a set of "negative" products for each user. These are randomly chosen
    // from among all of the other artists, excluding those that are "positive" for the user.
    val negativeData = testData.select(X, y).as[(Int,Int)].
      groupByKey { case (user, _) => user }.
      flatMapGroups { case (userID, userIDAndPosArtistIDs) =>
        val random = new Random()
        val posItemIDSet = userIDAndPosArtistIDs.map { case (_, item) => item }.toSet
        val negative = new ArrayBuffer[Int]()
        val allItems = B_allItems.value
        var i = 0
        // Make at most one pass over all artists to avoid an infinite loop.
        // Also stop when number of negative equals positive set size
        while (i < allItems.length && negative.size < posItemIDSet.size) {
          val artistID = allItems(random.nextInt(allItems.length))
          // Only add new distinct IDs
          if (!posItemIDSet.contains(artistID)) {
            negative += artistID
          }
          i += 1
        }
        // Return the set with user ID added back
        negative.map(artistID => (userID, artistID))
      }.toDF(X,y)

    // Make predictions on the rest:
    val negativePredictions = predictFunction(negativeData).
      withColumnRenamed("prediction", "negativePrediction")

    // Join positive predictions to negative predictions by user, only.
    // This will result in a row for every possible pairing of positive and negative
    // predictions within each user.
    val joinedPredictions = positivePredictions.join(negativePredictions, X).
      select(X, "positivePrediction", "negativePrediction").cache()

    // Count the number of pairs per user
    val allCounts = joinedPredictions.
      groupBy(X).agg(count(lit("1")).as("total")).
      select(X, "total")
    // Count the number of correctly ordered pairs per user
    val correctCounts = joinedPredictions.
      filter($"positivePrediction" > $"negativePrediction").
      groupBy(X).agg(count(X).as("correct")).
      select(X, "correct")

    // Combine these, compute their ratio, and average over all users
    val meanAUC = allCounts.join(correctCounts, Seq(X), "left_outer").
      select(col(X), (coalesce($"correct", lit(0)) / $"total").as("auc")).
      agg(mean("auc")).
      as[Double].first()

    joinedPredictions.unpersist()

    meanAUC
  }
}
