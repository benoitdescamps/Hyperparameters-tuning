name := "randomSearch"

version := "1.0"
scalaVersion := "2.11.8"
val sparkVersion = "2.3.0"

resolvers ++= Seq(
  //Spark
  "apache-snapshots" at "http://repository.apache.org/snapshots/",

  //Breeze
  "Sonatype Releases" at "https://oss.sonatype.org/content/repositories/releases/",

  //MMLSpark
  "MMLSpark Repo" at "https://mmlspark.azureedge.net/maven"
)

libraryDependencies ++= Seq(
  //Spark
  "org.apache.spark" %% "spark-core" % sparkVersion,
  "org.apache.spark" %% "spark-sql" % sparkVersion,
  "org.apache.spark" %% "spark-mllib" % sparkVersion,

  //Breeze
  "org.scalanlp" %% "breeze" % "0.13.2",
  "org.scalanlp" %% "breeze-natives" % "0.13.2",
  "org.scalanlp" %% "breeze-viz" % "0.13.2",

  //MMLSpark
  "com.microsoft.ml.spark" %% "mmlspark" % "0.12",

  "org.scalatest" %% "scalatest" % "2.2.6" % "test"
)

