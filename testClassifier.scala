//package org.apache.spark.ml.classification

import org.apache.spark.ml.util.{Identifiable, MetadataUtils}
import org.apache.spark.mllib.linalg.{SparseVector, DenseVector, Vector, Vectors}
import org.apache.spark.mllib.regression.LabeledPoint

import org.apache.spark.rdd.RDD
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{ProbabilisticClassifier}
import org.apache.spark.ml.param.ParamMap

import org.apache.spark.ml.feature.{StringIndexer, IndexToString, VectorIndexer}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.sql._


trait TestModel {
  override def toString: String = {
    s"TestModel"
  }

}

final class TestClassificationModel (
    override val uid: String,
    val numClasses: Int
  ) extends org.apache.spark.ml.classification.ProbabilisticClassificationModel[Vector, TestClassificationModel] with TestModel with Serializable {

    def this(numClasses: Int) =
      this(Identifiable.randomUID("testc"), numClasses)

    override def predictRaw(features: Vector): Vector = {
      val votes = Array.fill[Double](numClasses)(1.0/numClasses)
      Vectors.dense(votes)
    }

    override def predict(features: Vector): Double = {
      1.0/numClasses
    }

    override protected def raw2probabilityInPlace(rawPrediction: Vector): Vector = {
      rawPrediction match {
        case dv: DenseVector =>
          dv
        case sv: SparseVector =>
          throw new RuntimeException("Unexpected error in TestModel:" +
            " raw2probabilityInPlace encountered SparseVector")
      }
    }

    override def copy(extra: org.apache.spark.ml.param.ParamMap): TestClassificationModel = {
      new TestClassificationModel(numClasses)
    }

    override def toString: String = {
      s"TestModel (uid=$uid)"
    }
}

final class TestClassifier (
  override val uid: String) extends ProbabilisticClassifier[Vector, TestClassifier, TestClassificationModel] {

  def this() = this(Identifiable.randomUID("testc"))

  /*override def setSeed(value: Long): this.type = super.setSeed(value)*/

  override protected def train(dataset: DataFrame): TestClassificationModel = {
    new TestClassificationModel(uid, 2)
  }

  override def copy(extra: org.apache.spark.ml.param.ParamMap): TestClassifier = defaultCopy(extra)
}

/*
  Abstract MODEL 
*/



  

/*
object TestClassificationModel {

  /**
   * Normalize a vector of raw predictions to be a multinomial probability vector, in place.
   *
   * The input raw predictions should be >= 0.
   * The output vector sums to 1, unless the input vector is all-0 (in which case the output is
   * all-0 too).
   *
   * NOTE: This is NOT applicable to all models, only ones which effectively use class
   *       instance counts for raw predictions.
   */
  def normalizeToProbabilitiesInPlace(v: DenseVector): Unit = {
    val sum = v.values.sum
    if (sum != 0) {
      var i = 0
      val size = v.size
      while (i < size) {
        v.values(i) /= sum
        i += 1
      }
    }
  }
}
*/
