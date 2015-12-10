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


import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.{StringIndexer, IndexToString, VectorIndexer}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.sql._
import sqlContext.implicits._

case class Schema_Features(features: org.apache.spark.mllib.linalg.SparseVector)
case class Schema_TrainingSample(label: Double, features: org.apache.spark.mllib.linalg.SparseVector)

def toDF(x: org.apache.spark.rdd.RDD[Schema_TrainingSample]) = x.toDF()

def prepare_data(
    dataDF: org.apache.spark.sql.DataFrame, 
    featureIndices : List[Int]
  ): org.apache.spark.sql.DataFrame = {
  dataDF.rdd.map( sample => {
    val sparseFeatureVector = sample(1).asInstanceOf[org.apache.spark.mllib.linalg.SparseVector]
    val label = sample(0).asInstanceOf[Double]
    
    var values: Array[Double] = new Array[Double](featureIndices.size)
    for (i <- 0 to featureIndices.size-1) values(i) = sparseFeatureVector(featureIndices(i))
    
    val filteredFeatureVector = (new org.apache.spark.mllib.linalg.SparseVector(featureIndices.size, (0 to featureIndices.size-1).toArray, values)).toSparse
    
    Schema_TrainingSample(label, filteredFeatureVector)
  }).toDF()
}


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
      dataDF: org.apache.spark.sql.DataFrame,
      featureIndices: List[Int]
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
      dataDF: org.apache.spark.sql.DataFrame,
      featureIndices: List[Int]
      ): org.apache.spark.sql.DataFrame = {
    val predictions = pipelineModel.transform(prepare_data(dataDF, featureIndices))
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




/*
 * CLASSIFIER INTERFACE
 * 
 */

class CotrainingClassifier(
     pipelineFirst  : GenericPipelineForSSVLearning,
     pipelineSecond : GenericPipelineForSSVLearning, 
     maxIterations  : Int 
    ){		
  var labeledData : org.apache.spark.sql.DataFrame = _
  var featureIndices_1: List[Int] = _
  var featureIndices_2: List[Int] = _
  
  def fit(
    newLabeledData: org.apache.spark.sql.DataFrame, 
    newFeatureIndices_1: List[Int],
    newFeatureIndices_2: List[Int]): Unit = {
    labeledData = newLabeledData
    featureIndices_1 = newFeatureIndices_1
    featureIndices_2 = newFeatureIndices_2
  }  

  def predictData(dataDF: org.apache.spark.sql.DataFrame): org.apache.spark.sql.DataFrame = {
    //set initial sets
    var labeledData_1 = labeledData
    var labeledData_2 = labeledData
    var unlabeledData_1 = dataDF
    var unlabeledData_2 = dataDF
    //process initialization
    var numIterations = 0
    var progress: Long = 0
    //process
    do {
      pipelineFirst.fit(prepare_data(labeledData_1, featureIndices_1))
      pipelineSecond.fit(prepare_data(labeledData_2, featureIndices_2))
      val newData_1 = pipelineFirst.getNewReliableData(unlabeledData_1, featureIndices_1)
      val newData_2 = pipelineSecond.getNewReliableData(unlabeledData_2, featureIndices_2)
      labeledData_1 = labeledData_1.unionAll(newData_2)
      labeledData_2 = labeledData_2.unionAll(newData_1)
      unlabeledData_1 = unlabeledData_1.except(newData_2)
      unlabeledData_2 = unlabeledData_2.except(newData_1)
      progress = newData_1.count() + newData_2.count()
      numIterations += 1
    } while (numIterations < maxIterations && unlabeledData_1.count() > 0 && unlabeledData_2.count() > 0 && progress > 0)
    labeledData_1.intersect(labeledData_2)
  }
}





//load data
val data = MLUtils.loadLibSVMFile(sc, "data/a3a.txt").toDF()
var Array(labeledData, unlabeledData, testData) = data.randomSplit(Array(0.2, 0.5, 0.3))

//save to compare with learning without co-training
val originalLabeledData = labeledData

//classifiers initialization
val pipeline_1 = PipelineForSSVLearning("RandomForest", numTrees=5)
val pipeline_2 = PipelineForSSVLearning("RandomForest", numTrees=7)
pipeline_1.construct(prepare_data(data, (1 to 30).toList))
pipeline_2.construct(prepare_data(data, (31 to 60).toList))

val SSV_Classifier = new CotrainingClassifier(pipeline_1, pipeline_2, maxIterations=10)

SSV_Classifier.fit(labeledData, (1 to 30).toList, (31 to 60).toList)

val extraData = SSV_Classifier.predictData(unlabeledData)


//evaluate models and compare it with original classifier

//set evaluator
val evaluator = new MulticlassClassificationEvaluator().setLabelCol("indexedLabel").setPredictionCol("prediction").setMetricName("precision")

//set pipeline
val labelIndexer = new StringIndexer().setInputCol("label").setOutputCol("indexedLabel").fit(data)
val featureIndexer = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(5).fit(data)
val labelConverter = new IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(labelIndexer.labels)
val rf = new RandomForestClassifier().setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures").setNumTrees(30)
val pipeline = new Pipeline().setStages(Array(labelIndexer, featureIndexer, rf, labelConverter))
    
//set classifier on original data & evaluate it
val original_model = pipeline.fit(originalLabeledData)
val original_predictions = original_model.transform(testData)
val accuracy_origin = evaluator.evaluate(original_predictions)

//evaluate classifier on obtained data
val model = pipeline.fit(extraData)
val predictions = model.transform(testData)
val accuracy = evaluator.evaluate(predictions)

//ok done

val gain = accuracy-accuracy_origin
println("\nTOTAL:\nExtended Data size: " + extraData.count + "\nOriginal Data size: "+originalLabeledData.count+"\nTest Error = " + (1.0 - accuracy) +"\nTest Error Origin = " + (1.0 - accuracy_origin)+"\nGain: " + gain)

