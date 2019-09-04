package com.ajb0211.recommender.MovieLens

import scala.util.Random
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.recommendation.{ALS, ALSModel}
import org.apache.spark.sql.{DataFrame, Dataset, SparkSession}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._


object ALSRecommender {
  def load(modelPath: String)(spark: SparkSession,
                              ratingFile: String,
                              itemFile: String,
                              userFile: String,
                              partitions: Int = 4): ALSRecommender = {
    new ALSRecommender(spark, ratingFile, itemFile, userFile, partitions = partitions).loadModel(modelPath)
  }

  def fit(rank: Int,
          regParam: Double,
          alpha: Double,
          maxIter: Int
         )(spark: SparkSession,
           ratingFile: String,
           itemFile: String,
           userFile: String,
           partitions: Int = 4): ALSRecommender = {
    new ALSRecommender(spark, ratingFile, itemFile, userFile, partitions = partitions).fitModel(rank, regParam, alpha, maxIter)
  }
}



class ALSRecommender(
                      spark: SparkSession,
                      ratingFile: String,
                      itemFile: String,
                      userFile: String,
                      partitions: Int = 4,
                      var model: Option[ALSModel] = None
                    ) extends MovieLens(spark,ratingFile,itemFile,userFile,partitions) {
  // Cross joins are used for predictions
  spark.conf.set("spark.sql.crossJoin.enabled", "true")

  import ratingData.sparkSession.implicits._
  ratingData.createOrReplaceTempView("ratings")
  itemData.createOrReplaceTempView("items")

  def loadModel(modelPath: String): ALSRecommender = {model = Some(ALSModel.load(modelPath)); this}

  def fitModel(rank: Int,
               regParam: Double,
               alpha: Double,
               maxIter: Int
              ): ALSRecommender = {
    model = Some(initializeModel(rank,regParam,alpha,maxIter).fit(ratingData))
    this
  }

  def predictByUserID(userID: Int, nRecs: Int = 5): DataFrame = {
    val recFrame = model.get.itemFactors
      .select($"id".as("itemID")).
      withColumn("userID", lit(userID))
    val preds = model.get.transform(recFrame).
      select("itemID","prediction").
      orderBy($"prediction".desc).
      limit(nRecs)

//    preds.createOrReplaceTempView("preds")

    /* Use this query if we want predictions for unseen content */

//    spark.sql("""
//                SELECT * FROM preds
//                WHERE itemID NOT IN (
//                SELECT itemID FROM ratings
//                WHERE userID = 196)
//                """)

    preds.join(itemData,Seq("itemID"),"left").select("itemID","title","prediction")
  }

//  def predictByUserData(userData: DataFrame, nRecs: Int = 5): DataFrame = {
//
//  }



}
