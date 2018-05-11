/**
  * Created by bendesc on 11/05/2018.
  */

import breeze.stats.distributions.{Gamma, Gaussian}
import org.apache.spark.ml.classification.LogisticRegression

import tuning.RandomGridBuilder
object example_logistic_regression{
  def main(args: Array[String]): Unit = {

    val lr = new LogisticRegression()
      .setMaxIter(10)

    val randomGrid = new RandomGridBuilder(10)
      .addDistr(lr.regParam,Gamma(1.0,0.1))
      .addDistr(lr.elasticNetParam,Gamma(1.0,0.1))
      .addDistr(lr.threshold,Gaussian(0.5,0.05))
      .addDistr(lr.standardization,Array(true,false))
      .build()

    println(randomGrid.toList)
  }

}