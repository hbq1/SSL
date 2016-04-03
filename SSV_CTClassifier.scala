/*
 * Necessary to import SSV_Pipeline.scala before using
 */
 
import org.apache.spark.ml.classification.RandomForestClassifier

/*
 * Co-training model realization
 */

final class CoTrainingClassifier [
   FeatureType,
   E <: org.apache.spark.ml.classification.ProbabilisticClassifier[FeatureType, E, M], 
   M <: org.apache.spark.ml.classification.ProbabilisticClassificationModel[FeatureType, M]
] (
    val uid: String,
    val classifier1: E,
    val classifier2: E
  ) extends org.apache.spark.ml.classification.Classifier[FeatureType, CoTrainingClassifier[FeatureType,E,M], M] with Serializable {
  
  var countIterations: Long = 10
  var thresholdTrust: Double = 0.975
  var verbose: Boolean = false

  var data1: SSVData = _
  var data2: SSVData = _

  var featureIndices1: List[Int] = _
  var featureIndices2: List[Int] = _
  
  var pipelineSSV1 : PipelineForSSVLearning[FeatureType, E, M] = _
  var pipelineSSV2 : PipelineForSSVLearning[FeatureType, E, M] = _

  def this(classifier1: E, classifier2: E) =
      this(Identifiable.randomUID("ctc"), classifier1, classifier2)

  def setCountIterations(value: Long) = {
    countIterations = value
    this
  }

  def setThreshold(value: Double) = {
    thresholdTrust = value
    this
  }

  def setVerbose(value: Boolean) = {
    verbose = value
    this
  }

  def setFeaturesIndices(newFeaturesIndices1: List[Int], newFeaturesIndices2: List[Int]) = {
    featureIndices1 = newFeaturesIndices1
    featureIndices2 = newFeaturesIndices2
    this
  }

  def setFeaturesIndices1(newFeaturesIndices: List[Int]) = {
    featureIndices1 = newFeaturesIndices
    this
  }

  def setFeaturesIndices2(newFeaturesIndices: List[Int]) = {
    featureIndices2 = newFeaturesIndices
    this
  }
  
  def setPipeline(dataFull: DataFrame) = {
    pipelineSSV1 = new PipelineForSSVLearning[FeatureType, E, M](classifier1, thresholdTrust)
    pipelineSSV2 = new PipelineForSSVLearning[FeatureType, E, M](classifier2, thresholdTrust)
    val t_data = new SSVData()
    t_data.separateData(dataFull)
    pipelineSSV1.construct(t_data)
    pipelineSSV2.construct(t_data)
  }

  def train(labeledDataRaw: DataFrame, unlabeledDataRaw: DataFrame): M = {
    data1 = new SSVData()
    data2 = new SSVData()
    data1.prepareData(labeledDataRaw, unlabeledDataRaw)
    data2.prepareData(labeledDataRaw, unlabeledDataRaw)
    this.setPipeline(data1.mixData)
    this.train()
  }

  def train(dataset: DataFrame): M = {
    this.setPipeline(dataset)
    data1 = new SSVData()
    data2 = new SSVData()
    data1.separateData(dataset)
    data2.separateData(dataset)
    this.train()
  }

  def train(): M = {  
    if (verbose) {
      print("Entering train ...\n")
      print("Unlabeled count : " + data1.unlabeledData.count() + "\n")
      print("Labeled count : " + data1.labeledData.count() + "\n")
    }

    if (data1.unlabeledData.count > 0) {
      var numberOfIteration: Int = 0
      var countNewLabeled: Long = 0
      data1.setFeaturesIndices(featureIndices1)
      data2.setFeaturesIndices(featureIndices2)
      do {
        pipelineSSV1.fit(data1)
        pipelineSSV2.fit(data2)
        val newData1 = pipelineSSV1.getNewReliableData(data1)
        val newData2 = pipelineSSV2.getNewReliableData(data2)
        data1.rawLabeledData = data1.rawLabeledData.unionAll(newData2)
        data2.rawLabeledData = data2.rawLabeledData.unionAll(newData1)
        data1.rawUnlabeledData = data1.rawUnlabeledData.except(newData1.select("features"))
        data1.rawUnlabeledData = data1.rawUnlabeledData.except(newData2.select("features"))
        data2.rawUnlabeledData = data2.rawUnlabeledData.except(newData1.select("features"))
        data2.rawUnlabeledData = data2.rawUnlabeledData.except(newData2.select("features"))
        data1.setFeaturesIndices(featureIndices1)
        data2.setFeaturesIndices(featureIndices2)
        countNewLabeled = newData1.count() + newData2.count()
        if (verbose) {
          print("\nIteration " + (numberOfIteration+1) + "\n")
          print("New labeled count in first: " + newData1.count() + "\n")
          print("Labeled Data count in first: " + data1.labeledData.count() + "\n")
          print("New labeled count in second: " + newData2.count() + "\n")
          print("Labeled Data count in second: " + data2.labeledData.count() + "\n")
        }
        numberOfIteration += 1
      } while (numberOfIteration < countIterations && data1.unlabeledData.count() > 0 && data2.unlabeledData.count() > 0 && countNewLabeled > 0)
    }
    data1.labeledData = data1.rawLabeledData
    pipelineSSV1.constructModel(data1)
  }

  override def copy(extra: org.apache.spark.ml.param.ParamMap):CoTrainingClassifier[FeatureType,E,M] = defaultCopy(extra)
}


def CoTrainingRFClassifier(classifier1: RandomForestClassifier, classifier2: RandomForestClassifier, data: DataFrame) = {
  new CoTrainingClassifier[Vector, RandomForestClassifier, org.apache.spark.ml.classification.RandomForestClassificationModel](classifier1, classifier2)
}