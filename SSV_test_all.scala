/*
 * Necessary to import SSV_Pipeline.scala, SSV_SLClassifier.scala, SSV_CTClassifier.scala before usage
 */
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator


/*
 * Load data
 */
val data = MLUtils.loadLibSVMFile(sc, "data/a3a.txt").toDF()
var Array(labeledData, unlabeledDataX, testData) = data.randomSplit(Array(0.2, 0.5, 0.3))
var unlabeledData = unlabeledDataX.drop("label")
/*
 * Set Unlabeled data's labels = -1.0
 */
var allDirtyData = labeledData.unionAll(unlabeledData.withColumn("label", expr("-1.0")).select("label","features"))


/*
 * Train RandomForestClassifier
 */
val labelIndexerRF = new StringIndexer().setInputCol("label").setOutputCol("indexedLabel").fit(labeledData)
val RF = new RandomForestClassifier().setNumTrees(30).setFeaturesCol("features").setLabelCol("indexedLabel")
val labelConverterRF = new IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(labelIndexerRF.labels)
val pipelineRF = new Pipeline().setStages(Array(labelIndexerRF, RF, labelConverterRF))
var modelRF = pipelineRF.fit(labeledData)

/*
 * Train CoTrainingClassifier
 */
val labelIndexerCT = new StringIndexer().setInputCol("label").setOutputCol("indexedLabel").fit(allDirtyData)
val RF1 = new RandomForestClassifier().setNumTrees(30).setFeaturesCol("features").setLabelCol("indexedLabel")
val RF2 = new RandomForestClassifier().setNumTrees(30).setFeaturesCol("features").setLabelCol("indexedLabel")
val CT = CoTrainingRFClassifier(RF1, RF2, allDirtyData)
CT.setVerbose(true)
CT.setFeaturesIndices1((1 to 60).toList)
CT.setFeaturesIndices2((61 to 120).toList)
CT.setThreshold(0.97)
val labelConverterCT = new IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(labelIndexerCT.labels)
val pipelineCT = new Pipeline().setStages(Array(labelIndexerCT, CT, labelConverterCT))
var modelCT = pipelineCT.fit(allDirtyData)

/*
 * Train SelflearningClassifier
 */
val labelIndexerSL = new StringIndexer().setInputCol("label").setOutputCol("indexedLabel").fit(allDirtyData)
val RF3 = new RandomForestClassifier().setNumTrees(30).setFeaturesCol("features").setLabelCol("indexedLabel")
val SL = SelfLearningRFClassifier(RF3, allDirtyData)
SL.setVerbose(true)
SL.setThreshold(0.97)
val labelConverterSL = new IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(labelIndexerSL.labels)
val pipelineSL = new Pipeline().setStages(Array(labelIndexerSL, SL, labelConverterSL))
var modelSL = pipelineSL.fit(allDirtyData)

/*
 * Evaluate predictions 
 */
val evaluator = new MulticlassClassificationEvaluator().setLabelCol("indexedLabel").setPredictionCol("prediction").setMetricName("precision")

val predictionsRF = modelRF.transform(testData)
val accuracyRF = evaluator.setLabelCol("indexedLabel").setPredictionCol("prediction").evaluate(predictionsRF)

val predictionsCT = modelCT.transform(testData)
val accuracyCT = evaluator.setLabelCol("label").setPredictionCol("prediction").evaluate(predictionsCT)

val predictionsSL = modelSL.transform(testData)
val accuracySL = evaluator.setLabelCol("label").setPredictionCol("prediction").evaluate(predictionsSL)

/*
 * Result
 */
println("\nTOTAL:\nRF Error = " + (1.0 - accuracyRF) +"\nCT Error = " + (1.0 - accuracyCT)+"\nSL Error = " + (1.0 - accuracySL))
