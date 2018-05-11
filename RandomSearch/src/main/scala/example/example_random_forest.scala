/**
  * Created by bendesc on 11/05/2018.
  */

import breeze.stats.distributions.{NegativeBinomial, Poisson, Uniform}
import org.apache.spark.ml.classification.RandomForestClassifier

import tuning.RandomGridBuilder
object example_random_forest{
  def main(args: Array[String]): Unit = {
    val rf = new RandomForestClassifier()
    val randomGrid = new RandomGridBuilder(10)
      .addDistr(rf.numTrees,NegativeBinomial(150,0.6))
      .addDistr(rf.maxDepth,Poisson(8))
      .addDistr(rf.subsamplingRate,Uniform(0.7,0.99))
      .build()

    println(randomGrid.toList)
  }
}