/**
  * Created by bendesc on 11/05/2018.
  */

import breeze.stats.distributions.{Uniform, Poisson}
import org.apache.spark.ml.classification.GBTClassifier


import tuning.RandomGridBuilder
object example_gradient_boosting{
  def main(args: Array[String]): Unit = {
    val gbt = new GBTClassifier()
    val randomGrid = new RandomGridBuilder(10)
      .addDistr(gbt.maxIter,Poisson(200))
      .addDistr(gbt.maxDepth,Poisson(8))
      .addDistr(gbt.subsamplingRate,Uniform(0.7,0.99))
      .build()

    println(randomGrid.toList)
  }
}