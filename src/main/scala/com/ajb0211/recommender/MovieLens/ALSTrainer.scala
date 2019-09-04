package com.ajb0211.recommender.MovieLens

import java.io.{BufferedWriter, File, FileWriter}

import com.ajb0211.recommender.metrics.areaUnderCurve

import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.recommendation.{ALS, ALSModel}
import org.apache.spark.sql.{DataFrame, Dataset, SparkSession}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._


object ALSTrainer extends App {
  // If run locally, similar to Python:
  //   if __name__ == "__main__"

  val spark: SparkSession = SparkSession.
    builder.
    master("local[*]").
    appName("ALSTrainer").
    getOrCreate

  spark.sparkContext.setLogLevel("WARN")

  val dataDir: String = "./src/main/resources/ml-100k/"

  val alsVal: ALSTrainer = new ALSTrainer(spark,
    dataDir + "u.data",
    dataDir + "u.user",
    dataDir + "u.item").
    setRankRange(Seq(5,20)).
    setRegParamRange(Seq(0.001,0.1)).
    setAlphaRange(Seq(0.001,0.1)).
    setMaxIterRange(Seq(5,10)).
    setTrainFraction(0.85).
    setVerbose(true)


  val resources: String = "./src/main/resources/"

  val bestParams: alsVal.ScoredParams = alsVal.bestModel(
    printFile = Some(resources + "params.csv"),
    modelFile = Some(resources + "model")
  )

  println(bestParams)

}


class ALSTrainer(
  spark: SparkSession,
  ratingFile: String,
  itemFile: String,
  userFile: String,
  // use of vars allows using Mllib style "set" methods, at the expense of functional style points
  var rankRange: Iterable[Int] = List(5),
  var regParamRange: Iterable[Double] = List(0.01),
  var alphaRange: Iterable[Double] = List(0.1),
  var maxIterRange: Iterable[Int] = List(20),
  var trainFraction: Double = 0.85,
  var trainFile: Option[String] = None,
  var verbose: Boolean = false,
  partitions: Int = 4
) extends MovieLens(spark,ratingFile,itemFile,userFile,partitions) {

  // would rather access spark object but throwing error, not understood at this time
  import itemData.sparkSession.implicits._

  type ScoredParams = (Double, (Int,Double,Double,Int))

  val allItems: Array[Int] = itemData.select("itemID").as[Int].distinct.collect
  val b_allItems :Broadcast[Array[Int]] = spark.sparkContext.broadcast(allItems)

  // Return of this for method chaining
  // Arguments are initialisms
  // Future work: unpack dictionary for parameter setting
  def setRankRange(rr: Iterable[Int]): ALSTrainer = {rankRange = rr; this}
  def setRegParamRange(rpr: Iterable[Double]): ALSTrainer = {regParamRange = rpr; this}
  def setAlphaRange(ar: Iterable[Double]): ALSTrainer = {alphaRange = ar; this}
  def setMaxIterRange(mir: Iterable[Int]): ALSTrainer = {maxIterRange = mir; this}
  def setTrainFraction(tf: Double): ALSTrainer = {trainFraction = tf; this}
  def setVerbose(v: Boolean): ALSTrainer = {verbose = v; this}
  def setTrainFile(tf: String): ALSTrainer = {trainFile = Some(tf); this}



  def validate(): Seq[ScoredParams] = {
    val Array(trainData, testData) = ratingData.randomSplit(Array(trainFraction,1.0-trainFraction))

    // Use map operations to conditionally print
    val file = trainFile.map(new File(_))
    val bufferedWriter = file.map{ f => new BufferedWriter(new FileWriter(f))}
    bufferedWriter.map{ bw =>
      bw.write("auc,rank,reg,alpha,maxIter\n")
      None
    }

    val out = for (
      rank <- rankRange;
      reg <- regParamRange;
      alpha <- alphaRange;
      maxIter <- maxIterRange
    ) yield {

      val model: ALSModel= initializeModel(rank,reg,alpha,maxIter).fit(trainData)

      val auc = areaUnderCurve("userID","itemID",testData, b_allItems, model.transform)

      // free resources, do not wait for garbage collection after model is deleted
      model.userFactors.unpersist
      model.itemFactors.unpersist
      if (verbose){
        println((auc, (rank,reg,alpha,maxIter)))
      }

      // Return None for type-check
      bufferedWriter.map{f => f.write(s"$auc,$rank,$reg,$alpha,$maxIter\n");None}
      (auc, (rank,reg,alpha,maxIter))
    }

    bufferedWriter.map{bw => bw.close();None}

    out
  }.toSeq

  // evaluate when needed
  // iteration: make this resettable => model object reusable
  lazy val scores: Seq[ScoredParams] = validate()

  def bestModel(
                printFile: Option[String] = None,
                modelFile: Option[String] = None
               ): ScoredParams = {

    // Sort key-value pairs by key in ascending order, then reverse and take first element
    val (auc, (rank,reg,alpha,maxIter)) = scores.sorted.reverse.head

    // Use Scala handling of monadic types, None will cause map to return None
    printFile.map{ fileName =>
      val file = new File(fileName)
      val bufferedWriter = new BufferedWriter(new FileWriter(file))
      bufferedWriter.write(s"$rank,$reg,$alpha,$maxIter")  // String interpolation
      bufferedWriter.close()
      None // For type-checking, no consequence
    }

    modelFile.map{ fileName =>
      val model = initializeModel(rank,reg,alpha,maxIter).fit(ratingData)
      model.write.overwrite.save(fileName)
      None
    }

    (auc, (rank,reg,alpha,maxIter))
  }




}
