package tuning

import scala.collection.mutable
import org.apache.spark.ml.param._
import breeze.stats.distributions
import breeze.stats.distributions.Rand

import scala.tools.scalap.scalax.rules.Error
/**
  * Created by bendesc on 10/05/2018.
  */



class RandomGridBuilder(n: Int) {

  private val paramDistr = mutable.Map.empty[Param[_],Any]

  def addDistr[T](param: Param[T], distr: Any ): this.type = distr match {
    case _ : Rand[_] => {paramDistr.put(param, distr)
      this}
    case _ : Array[_] => { paramDistr.put(param, distr)
      this}
    case _  => throw new NotImplementedError("Distribution should be of type breeze.stats.distributions.Rand or an Array")
}
  /**
    * Similar to GridSearch
    * Builds and returns all combinations of parameters specified by the param grid.
    */
   def build():  Array[ParamMap] = {
     var paramMaps = (1 to n).map( _ => new ParamMap())

     paramDistr.foreach{
       case (param, distribution) =>

         val values = distribution match {
             case d :Rand[_] => {
               paramMaps.map(_.put(param.asInstanceOf[Param[Any]],d.sample()))
             }
             case d: Array[_] => {
               val r = scala.util.Random
               paramMaps.map(_.put(param.asInstanceOf[Param[Any]], d(r.nextInt(d.length))) )
             }
         }
     }
     paramMaps.toArray
   }

}


