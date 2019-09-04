package com.ajb0211.recommender

import org.apache.spark.SparkContext

import scala.util.Random
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.recommendation.{ALS, ALSModel}
import org.apache.spark.sql.{DataFrame, Dataset, SparkSession}
import org.apache.spark.sql.functions._

object test extends App{
  val spark: SparkSession = SparkSession.
      builder.
      master("local[*]").
      appName("test").
      getOrCreate

  val sc: SparkContext = spark.sparkContext

  sc.setLogLevel("WARN")

  val testRDD = sc.parallelize(1 to 1000, 4)

  testRDD.collect.foreach(println)
}
