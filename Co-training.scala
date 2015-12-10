import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.feature.{StringIndexer, IndexToString, VectorIndexer}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.sql._
import sqlContext.implicits._

case class Schema_Features(features: org.apache.spark.mllib.linalg.SparseVector)
case class Schema_TrainingSample(label: Double, features: org.apache.spark.mllib.linalg.SparseVector)

def toDF(x: org.apache.spark.rdd.RDD[Schema_TrainingSample]) = x.toDF()

def prepare_data(x: org.apache.spark.sql.DataFrame, feature_indices_from: Int, feature_indices_to: Int): org.apache.spark.sql.DataFrame = {
  x.rdd.map( r => {
    val sparse_v = r(1).asInstanceOf[org.apache.spark.mllib.linalg.SparseVector]
    val indices = sparse_v.indices.filter(i => i >= feature_indices_from && i < feature_indices_to)
    val from = if (indices.size>0) sparse_v.indices.indexOf(indices(0)) else 0
    val values = sparse_v.values.slice(from, from + indices.size)
    val size: Int = feature_indices_to - feature_indices_from
    val res_vector = new org.apache.spark.mllib.linalg.SparseVector(size, indices.map(x=>x-feature_indices_from), values)
    Schema_TrainingSample(r(0).asInstanceOf[Double], res_vector)
  }).toDF()
}

class Classifier(feature_indices_from: Int, feature_indices_to: Int, threshold: Double) {
  var model: org.apache.spark.ml.PipelineModel = _
  var pipeline: org.apache.spark.ml.Pipeline  = _

  def get_pipeline(x: org.apache.spark.sql.DataFrame) : org.apache.spark.ml.Pipeline = {
    val labelIndexer = new StringIndexer().setInputCol("label").setOutputCol("indexedLabel").fit(x)
    val featureIndexer = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(5).fit(x)
    val labelConverter = new IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(labelIndexer.labels)
    val rf = new RandomForestClassifier().setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures").setNumTrees(5)
    new Pipeline().setStages(Array(labelIndexer, featureIndexer, rf, labelConverter))
  }

  def fit(x: org.apache.spark.sql.DataFrame) = {
    val df = prepare_data(x, feature_indices_from, feature_indices_to)
    model = pipeline.fit(df)
  }

  def precalc_pipeline(x: org.apache.spark.sql.DataFrame) = {
	pipeline = get_pipeline(prepare_data(x, feature_indices_from, feature_indices_to))
  }

  def predict(x: org.apache.spark.sql.DataFrame): org.apache.spark.sql.DataFrame = {
    model.transform(x) 
  }

  def get_new_reliable_data(x: org.apache.spark.sql.DataFrame): org.apache.spark.sql.DataFrame = {
    val predictions = predict(prepare_data(x, feature_indices_from, feature_indices_to))
    val predictions_probability = predictions.select("probability").rdd.map( r => {
      r(0).asInstanceOf[org.apache.spark.mllib.linalg.DenseVector].toArray.zipWithIndex.maxBy(_._1)
    }).collect()
    val features = x.select("features").rdd.zipWithIndex.collect()
    toDF(sc.parallelize(features.filter(z => predictions_probability(z._2.toInt)._1 > threshold)
                                .map(x => Schema_TrainingSample(predictions_probability(x._2.toInt)._2, x._1(0).asInstanceOf[org.apache.spark.mllib.linalg.SparseVector]))))
  }
}



//load data
val data = MLUtils.loadLibSVMFile(sc, "data/a3a.txt").toDF()

var Array(labeledData, unlabeledData, testData) = data.randomSplit(Array(0.2, 0.5, 0.3))

//set initial sets
var labeled_data_1 = labeledData
var labeled_data_2 = labeledData
var unlabeled_data_1 = unlabeledData
var unlabeled_data_2 = unlabeledData

//save to compare with learning without co-training
val originalLabeledData = labeledData

//pipeline initialization



//classifiers initialization
val Classifier_1 = new Classifier(0, 60, .95)
val Classifier_2 = new Classifier(60, 123, .95)
Classifier_1.precalc_pipeline(data)
Classifier_2.precalc_pipeline(data)

//process initialization
var numIterations = 0
val maxIterations = 15
var progress: Long = 0

//process
do {
  Classifier_1.fit(labeled_data_1)
  Classifier_2.fit(labeled_data_2)
  val new_data_1 = Classifier_1.get_new_reliable_data(unlabeled_data_1)
  val new_data_2 = Classifier_2.get_new_reliable_data(unlabeled_data_2)
  
  labeled_data_1 = labeled_data_1.unionAll(new_data_2)
  labeled_data_2 = labeled_data_2.unionAll(new_data_1)

  unlabeled_data_1 = unlabeled_data_1.except(new_data_2)
  unlabeled_data_2 = unlabeled_data_2.except(new_data_1)

  progress = new_data_1.count() + new_data_2.count()
  numIterations += 1
} while (numIterations < maxIterations && unlabeled_data_1.count() > 0 && unlabeled_data_2.count() > 0 && progress > 0)

val commonReliableData = labeled_data_1.intersect(labeled_data_2)

//___evaluate models and compare it with original classifier

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
val model = pipeline.fit(commonReliableData)
val predictions = model.transform(testData)
val accuracy = evaluator.evaluate(predictions)

//ok done

val gain = accuracy-accuracy_origin
println("\nTOTAL:\nCommonData size: " + commonReliableData.count + "\nOriginalData size: "+originalLabeledData.count+"\nTest Error = " + (1.0 - accuracy) +"\nTest Error Origin = " + (1.0 - accuracy_origin)+"\nIterations: " + numIterations+"\nGain: " + gain)

