/// /home/yuri/Desktop/SSL/SSVPipeline.scala

//package SSVPipeline


import org.apache.spark.ml.util.{Identifiable, MetadataUtils}
import org.apache.spark.mllib.linalg.{SparseVector, DenseVector, Vector, Vectors}
import org.apache.spark.mllib.regression.LabeledPoint

import org.apache.spark.ml.feature.{StringIndexer, IndexToString, VectorIndexer}
import org.apache.spark.ml.Pipeline

import org.apache.spark.sql._
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._


///////////////COMMON INTERFACE ///////////////////////////

/*
 *	Schema for ML pipeline 
 */
case class SchemaTrainingSample(label: Double, features: org.apache.spark.mllib.linalg.SparseVector)

/*
 *  Transformer: Schema => DataFrame
 */
def toDF(x: org.apache.spark.rdd.RDD[SchemaTrainingSample]): org.apache.spark.sql.DataFrame  = x.toDF()


/* 
 * Class to prepare and operate on data
 */
class SSVData extends Serializable {
  /*
   *  Filter data and extract only necessary fields
   */  

  var rawLabeledData : org.apache.spark.sql.DataFrame  = _
  var rawUnlabeledData : org.apache.spark.sql.DataFrame  = _
   
  var labeledData : org.apache.spark.sql.DataFrame  = _
  var unlabeledData : org.apache.spark.sql.DataFrame = _
  
  def prepareData(labeledData : org.apache.spark.sql.DataFrame, unlabeledData : org.apache.spark.sql.DataFrame): SSVData = {
    this.rawLabeledData = labeledData.select("label", "features")
    this.rawUnlabeledData = unlabeledData.select("features")
    this.labeledData = this.rawLabeledData
    this.unlabeledData = this.rawUnlabeledData
    this
  }

  def setFeaturesIndices(indices: List[Int]): SSVData = {
    labeledData = toDF(rawLabeledData.rdd.map( sample => {
      val sparseFeatureVector = sample(1).asInstanceOf[org.apache.spark.mllib.linalg.SparseVector]
      val label = sample(0).asInstanceOf[Double]
      var values: Array[Double] = new Array[Double](indices.size)
      for (i <- 0 to indices.size-1) values(i) = sparseFeatureVector(indices(i))
    
      val filteredFeatureVector = (new org.apache.spark.mllib.linalg.SparseVector(indices.size, (0 to indices.size-1).toArray, values)).toSparse
    
      SchemaTrainingSample(label, filteredFeatureVector)
    }))
    unlabeledData = toDF(rawUnlabeledData.rdd.map( sample => {
      val sparseFeatureVector = sample(0).asInstanceOf[org.apache.spark.mllib.linalg.SparseVector]
      var values: Array[Double] = new Array[Double](indices.size)
      for (i <- 0 to indices.size-1) values(i) = sparseFeatureVector(indices(i))
    
      val filteredFeatureVector = (new org.apache.spark.mllib.linalg.SparseVector(indices.size, (0 to indices.size-1).toArray, values)).toSparse
      val label = (-1).asInstanceOf[Double]
      SchemaTrainingSample(label, filteredFeatureVector)
    })).drop("label")
    this
  }
  
  /*
   * For testing, just to mix labeled and unlabeled data
   */ 
   
  def mixData:org.apache.spark.sql.DataFrame = {
    rawLabeledData.unionAll(rawUnlabeledData.withColumn("label", expr("-1.0")).select("label","features"))
  }

  /* Separate labeled & unlabeled data
   * Assume that unlabeled data have label < 0
   */
   
  def separateData(data : org.apache.spark.sql.DataFrame): SSVData = {
    this.rawLabeledData = data.filter(data("label") >= 0).select("label", "features")
    this.rawUnlabeledData = data.filter(data("label") < 0).select("features")
    this.labeledData = this.rawLabeledData
    this.unlabeledData = this.rawUnlabeledData
    this
  }
 
}

/*
 * 	Pipeline for semi-supervised learning
 * Acually, it is a wrapper for ml.Pipeline, extended with special methods for ssv-learning
 * It gets classifier and threshold of acceptance.
 * Used for Classifiers optimisation - once constructed, always used
 */
class PipelineForSSVLearning [
    FeatureType,
    E <: org.apache.spark.ml.classification.ProbabilisticClassifier[FeatureType, E, M], 
    M <: org.apache.spark.ml.classification.ProbabilisticClassificationModel[FeatureType, M]
   ](
     val classifier : E,
     val thresholdTrust : Double
    ) extends Serializable {
  var pipelineModel: org.apache.spark.ml.PipelineModel = _
  var pipeline:   org.apache.spark.ml.Pipeline = _

  var labelIndexer: org.apache.spark.ml.feature.StringIndexerModel = _
  var featureIndexer: org.apache.spark.ml.feature.VectorIndexerModel = _
  var labelConverter: org.apache.spark.ml.feature.IndexToString = _

 /*
  * Pipeline construction and label's indexing on stages
    Associated with classifier
  */
  def construct(data : SSVData): org.apache.spark.ml.Pipeline = {
    labelIndexer = new StringIndexer().setInputCol("label").setOutputCol("indexedLabel").fit(data.labeledData)
    labelConverter = new IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(labelIndexer.labels)
    pipeline = new Pipeline()
    pipeline.setStages(Array(labelIndexer, classifier.setLabelCol("indexedLabel").setFeaturesCol("features"), labelConverter))
    pipeline
  }
  
  def constructModel(data : SSVData): M = {
    classifier.fit(labelIndexer.transform(data.labeledData))
  }
  
 /*
  * Fit the pipeline model
  */  
  def fit(data : SSVData): org.apache.spark.ml.PipelineModel = {
    pipelineModel = pipeline.fit(data.labeledData)
    pipelineModel
  }
  
 /*
  * Predict labels (in other words, transform data)
  */ 
  def transform(data : SSVData): org.apache.spark.sql.DataFrame = {
    pipelineModel.transform(data.unlabeledData)
  }

 /*
  * Special method for ssv-learning.
  * It predicts labels and returns samples which classifier defined as reliable (their predicted probability > thresholdTrust)
  * Returns DataFrame of reliable samples
  */  
  
  def getNewReliableData(
      data: SSVData
      ): org.apache.spark.sql.DataFrame = {
    val predictions = pipelineModel.transform(data.unlabeledData)
    val predictionsProbability = predictions.select("probability").rdd.map( r => {
      r(0).asInstanceOf[org.apache.spark.mllib.linalg.DenseVector].toArray.zipWithIndex.maxBy(_._1)
    }).collect()
    val features = data.rawUnlabeledData.select("features").rdd.zipWithIndex.collect()
    toDF(sc.parallelize(features.filter(z => predictionsProbability(z._2.toInt)._1 > thresholdTrust)
                                .map(x => SchemaTrainingSample(predictionsProbability(x._2.toInt)._2, x._1(0)
                                          .asInstanceOf[org.apache.spark.mllib.linalg.SparseVector])
                                     )
                        ))
  }

  def copy(): PipelineForSSVLearning[FeatureType, E, M] = {
    val result = new PipelineForSSVLearning[FeatureType,E,M](classifier, thresholdTrust)
    result.pipelineModel = pipelineModel
  	result.pipeline = pipeline
  	result.labelIndexer = labelIndexer
  	result.labelConverter = labelConverter
  	result
  }
}
