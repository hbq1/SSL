import org.apache.spark.annotation.{Experimental, Since}
import org.apache.spark.ml.tree.impl.RandomForest
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.tree.{DecisionTreeModel, RandomForestParams, TreeClassifierParams, TreeEnsembleModel}
import org.apache.spark.ml.util.{Identifiable, MetadataUtils}
import org.apache.spark.mllib.linalg.{SparseVector, DenseVector, Vector, Vectors}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.configuration.{Algo => OldAlgo}
import org.apache.spark.mllib.tree.model.{RandomForestModel => OldRandomForestModel}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.classification.DecisionTreeClassificationModel
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.feature.{StringIndexer, IndexToString, VectorIndexer}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.sql._

case class Schema_TrainingSample(label: Double, features: org.apache.spark.mllib.linalg.SparseVector)

def toDF(x: org.apache.spark.rdd.RDD[Schema_TrainingSample]) = x.toDF()
/*
 * PIPELINES FOR CLASSIFIER
 * 
 * 
 */
 
abstract class GenericPipelineForSSVLearning() {
  def construct(dataDF : org.apache.spark.sql.DataFrame): Unit 
  def fit(dataDF : org.apache.spark.sql.DataFrame): org.apache.spark.ml.PipelineModel 
  def transform(dataDF : org.apache.spark.sql.DataFrame): org.apache.spark.sql.DataFrame 
  def getNewReliableData(
      dataDF: org.apache.spark.sql.DataFrame
      ): org.apache.spark.sql.DataFrame 
} 

class PipelineForSSVLearning_Template [
     FeatureType,
     E <: org.apache.spark.ml.classification.ProbabilisticClassifier[FeatureType, E, M], 
     M <: org.apache.spark.ml.classification.ProbabilisticClassificationModel[FeatureType, M]
    ](
     classifierModel : E,
     thresholdTrust : Double
    ) extends GenericPipelineForSSVLearning {
  var pipelineModel: org.apache.spark.ml.PipelineModel = _
  var pipelineRaw:   org.apache.spark.ml.Pipeline = _

  def construct(dataDF : org.apache.spark.sql.DataFrame): Unit = {
    val labelIndexer = new StringIndexer().setInputCol("label").setOutputCol("indexedLabel").fit(dataDF)
    val featureIndexer = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(5).fit(dataDF)
    val labelConverter = new IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(labelIndexer.labels)
    pipelineRaw = new Pipeline().setStages(Array(labelIndexer, featureIndexer, classifierModel, labelConverter))
  }
  
  def fit(dataDF : org.apache.spark.sql.DataFrame): org.apache.spark.ml.PipelineModel = {
    pipelineModel = pipelineRaw.fit(dataDF)
    pipelineModel
  }
  
  def transform(dataDF : org.apache.spark.sql.DataFrame): org.apache.spark.sql.DataFrame = {
    pipelineModel.transform(dataDF)
  }
  
  def getNewReliableData(
      dataDF: org.apache.spark.sql.DataFrame
      ): org.apache.spark.sql.DataFrame = {
    val predictions = pipelineModel.transform(dataDF)
    val predictionsProbability = predictions.select("probability").rdd.map( r => {
      r(0).asInstanceOf[org.apache.spark.mllib.linalg.DenseVector].toArray.zipWithIndex.maxBy(_._1)
    }).collect()
    val features = dataDF.select("features").rdd.zipWithIndex.collect()
    toDF(sc.parallelize(features.filter(z => predictionsProbability(z._2.toInt)._1 > thresholdTrust)
                                .map(x => Schema_TrainingSample(predictionsProbability(x._2.toInt)._2, x._1(0)
                                          .asInstanceOf[org.apache.spark.mllib.linalg.SparseVector])
                                     )
                        )
     )
  }
} 
 
def PipelineForSSVLearning(
    classificator: String = "RandomForest", 
    numTrees: Int = 10,
    thresholdTrust: Double = 0.95
    ): GenericPipelineForSSVLearning = {
  if (classificator == "RandomForest") {
    val RF = new RandomForestClassifier().setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures").setNumTrees(numTrees)
    new PipelineForSSVLearning_Template[Vector, RandomForestClassifier, org.apache.spark.ml.classification.RandomForestClassificationModel](RF, thresholdTrust)
  } else {
    //TODO: add new probabilistic models
    null
  }
}


////////////////////

class SelflearningClassifier(
     pipeline  : GenericPipelineForSSVLearning,
     maxIterations  : Int 
    ){		
  var labeledData : org.apache.spark.sql.DataFrame = _

  
  def fit(newLabeledData: org.apache.spark.sql.DataFrame): Unit = {
    labeledData = newLabeledData
  }  

  def predictData(dataDF: org.apache.spark.sql.DataFrame): org.apache.spark.sql.DataFrame = {
    var unlabeledData = dataDF 
    
    var numIterations: Int = 0
    var progress: Long = 0

    do {
      val model = pipeline.fit(labeledData)
      val newData = pipeline.getNewReliableData(unlabeledData)
      progress = newData.count
      unlabeledData = unlabeledData.except(newData)
      labeledData = labeledData.unionAll(newData)
      numIterations += 1
    } while (numIterations < maxIterations && unlabeledData.count > 0 && progress > 0)
    labeledData
  }
}


val data = MLUtils.loadLibSVMFile(sc, "data/a3a.txt").toDF()
var Array(labeledData, unlabeledData, testData) = data.randomSplit(Array(0.2, 0.5, 0.3))

//classifiers initialization
val pipeline_ = PipelineForSSVLearning("RandomForest", numTrees=25)

pipeline_.construct(data)

val SSV_Classifier = new SelflearningClassifier(pipeline_, maxIterations=10)

SSV_Classifier.fit(labeledData)

val extraData = SSV_Classifier.predictData(unlabeledData)

//set evaluator
val evaluator = new MulticlassClassificationEvaluator().setLabelCol("indexedLabel").setPredictionCol("prediction").setMetricName("precision")

//set pipeline
val labelIndexer = new StringIndexer().setInputCol("label").setOutputCol("indexedLabel").fit(data)
val featureIndexer = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(5).fit(data)
val labelConverter = new IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(labelIndexer.labels)
val rf = new RandomForestClassifier().setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures").setNumTrees(30)
val pipeline = new Pipeline().setStages(Array(labelIndexer, featureIndexer, rf, labelConverter))
    
//set classifier on original data & evaluate it
val original_model = pipeline.fit(labeledData)
val original_predictions = original_model.transform(testData)
val accuracy_origin = evaluator.evaluate(original_predictions)

//evaluate classifier on obtained data
val model = pipeline.fit(extraData)
val predictions = model.transform(testData)
val accuracy = evaluator.evaluate(predictions)

//ok done

val gain = accuracy-accuracy_origin
println("\nTOTAL:\nExtended Data size: " + extraData.count + "\nOriginal Data size: "+labeledData.count+"\nTest Error = " + (1.0 - accuracy) +"\nTest Error Origin = " + (1.0 - accuracy_origin)+"\nGain: " + gain)

