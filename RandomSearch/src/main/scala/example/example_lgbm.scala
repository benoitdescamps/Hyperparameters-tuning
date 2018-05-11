/**
  * Created by bendesc on 11/05/2018.
  */

import breeze.stats.distributions.{Gamma,Uniform,Poisson}
import com.microsoft.ml.spark.LightGBMClassifier
/**
  * Created by bendesc on 10/05/2018.
  */

import tuning.RandomGridBuilder
object example_lgbm{
  def main(args: Array[String]): Unit = {

    val lgbm = new LightGBMClassifier()

    //lgbm.re
    val randomGrid = new RandomGridBuilder(5)
      .addDistr(lgbm.learningRate,Gamma(1.0,0.1))
      .addDistr(lgbm.baggingFraction,Uniform(0.7,0.99))
      .addDistr(lgbm.featureFraction,Uniform(0.7,0.99))
      .addDistr(lgbm.maxDepth,Poisson(10))
      .addDistr(lgbm.numIterations,Poisson(200))
      .build()

    println(randomGrid.toList)
  }

}