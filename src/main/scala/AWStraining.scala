import com.ajb0211.recommender.MovieLens.ALSTrainer
import org.apache.spark.sql.SparkSession
import scala.util.Random
import scala.math.pow

object AWStraining extends App{
  val spark: SparkSession = SparkSession.
    builder.
    master("local[*]").
    appName("MovieLensParamSearch").
    getOrCreate

  spark.sparkContext.setLogLevel("WARN")

  val dataDir: String = "./src/main/resources/ml-100k/"

  val rnd = new Random(0)

  val ranks: Seq[Int] = 10 to 500 by 200
  val regs: Seq[Double] = (-300 to 200 by 200).map(x => pow(10,x/100))
  val alphas: Seq[Double] = (-500 to 200 by 200).map(x => pow(10,x/100))

  val alsVal: ALSTrainer = new ALSTrainer(spark,
    dataDir + "u.data",
    dataDir + "u.user",
    dataDir + "u.item").
    setRankRange(ranks).
    setRegParamRange(regs).
    setAlphaRange(alphas).
    setMaxIterRange(Seq(5)).
    setTrainFraction(0.85).
    setVerbose(true).
    setTrainFile("./src/main/resources/trainHistory.csv")



  val resources: String = "./src/main/resources/"

  val bestParams: alsVal.ScoredParams = alsVal.bestModel(
    printFile = Some(resources + "params.csv"),
    modelFile = Some(resources + "model")
  )

  println(bestParams)
}
