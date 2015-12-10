import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.classification.DecisionTreeClassificationModel
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.feature.{StringIndexer, IndexToString, VectorIndexer}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.sql._



val data = MLUtils.loadLibSVMFile(sc, "data/mllib/sample_libsvm_data.txt").toDF()


val labelIndexer = new StringIndexer().setInputCol("label").setOutputCol("indexedLabel").fit(data)
val featureIndexer = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(4).fit(data)
//val dt = new DecisionTreeClassifier().setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures")
val rf = new RandomForestClassifier().setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures").setNumTrees(300)
val labelConverter = new IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(labelIndexer.labels)

val pipeline = new Pipeline().setStages(Array(labelIndexer, featureIndexer, rf, labelConverter))
//val pipeline = new Pipeline().setStages(Array(labelIndexer, featureIndexer, dt, labelConverter))

case class Schema_Features(features: org.apache.spark.mllib.linalg.SparseVector)
case class Schema_TrainingSample(label: Double, features: org.apache.spark.mllib.linalg.SparseVector)


var Array(labeledData, unlabeledDataX, testData) = data.randomSplit(Array(0.2, 0.5, 0.3))


val originalLabeledData = labeledData
var unlabeledData = unlabeledDataX.drop("label").rdd 
var numIterations = 0


val maxIterations = 15
val threshold = 0.95
var progress: Long = 0

do {	
	var model = pipeline.fit(labeledData)

	val predictions = model.transform(unlabeledData.map(x => Schema_Features(x(0).asInstanceOf[org.apache.spark.mllib.linalg.SparseVector])).toDF())
	val predictions_probability = predictions.select("probability").rdd.map( r => {
			r(0).asInstanceOf[org.apache.spark.mllib.linalg.DenseVector].toArray.zipWithIndex.maxBy(_._1)
		}).collect()
	val features = predictions.select("features").rdd.zipWithIndex
	
	val reliableData = features.filter(x => predictions_probability(x._2.toInt)._1 > threshold).map(x => Schema_TrainingSample(predictions_probability(x._2.toInt)._2, x._1(0).asInstanceOf[org.apache.spark.mllib.linalg.SparseVector])).toDF()
	progress = reliableData.count
	
	//print(progress)
	
	unlabeledData = features.filter(x => predictions_probability(x._2.toInt)._1 <= threshold).map(x => x._1)
	labeledData = labeledData.unionAll(reliableData)
	numIterations += 1
} while (numIterations < maxIterations && unlabeledData.count > 0 && progress > 0)

val evaluator = new MulticlassClassificationEvaluator().setLabelCol("indexedLabel").setPredictionCol("prediction").setMetricName("precision")

val original_model = pipeline.fit(originalLabeledData)
val origin_predictions = original_model.transform(testData)

val accuracy_origin = evaluator.evaluate(origin_predictions)


val model = pipeline.fit(labeledData)
val predictions = model.transform(testData)

val accuracy = evaluator.evaluate(predictions)


val gain = accuracy-accuracy_origin

println("Test Error = " + (1.0 - accuracy))
println("Test Error Origin = " + (1.0 - accuracy_origin))
println("Iterations: " + numIterations)
println("Gain: " + gain)

