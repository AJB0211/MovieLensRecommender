name := "MovieLensRecommender"

version := "1.0"

scalaVersion := "2.12.8"

organization := "com.ajb0211"



libraryDependencies ++= Seq("org.apache.spark" %% "spark-core" % "2.4.3",
                            "org.apache.spark" %% "spark-streaming" % "2.4.3",
                            "org.apache.spark" %% "spark-sql" % "2.4.3",
                            "org.apache.spark" %% "spark-mllib" % "2.4.3"
)