/*
 * Necessary to import SSV_Pipeline.scala before usage
 */
import org.apache.spark.ml.classification.RandomForestClassifier


/*
 * Self-learning model realization
 */

final class SelfLearningClassifier [
   FeatureType,
   E <: org.apache.spark.ml.classification.ProbabilisticClassifier[FeatureType, E, M], 
   M <: org.apache.spark.ml.classification.ProbabilisticClassificationModel[FeatureType, M]
] (
    val uid: String,
    val classifier: E
  ) extends org.apache.spark.ml.classification.Classifier[FeatureType, SelfLearningClassifier[FeatureType,E,M], M] with Serializable {
  
  var countIterations: Long = 10
  var thresholdTrust: Double = 0.975
  var data: SSVData = _
  var verbose: Boolean = false
  
  var pipelineSSV : PipelineForSSVLearning[FeatureType, E, M] = _

  def this(classifier: E) =
      this(Identifiable.randomUID("slc"), classifier)

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

  def setPipeline(data: SSVData) = {
    pipelineSSV = new PipelineForSSVLearning[FeatureType, E, M](classifier, thresholdTrust)
    pipelineSSV.construct(data)
  }
  
  def setPipeline(dataFull: DataFrame) = {
    val t_data = new SSVData()
    t_data.separateData(dataFull)
    pipelineSSV = new PipelineForSSVLearning[FeatureType, E, M](classifier, thresholdTrust)
    pipelineSSV.construct(t_data)
  }

  def train(labeledDataRaw: DataFrame, unlabeledDataRaw: DataFrame): M = {
    data = new SSVData()
    data.prepareData(labeledDataRaw, unlabeledDataRaw)
    this.setPipeline(data)
    this.train()
  }

  def train(dataset: DataFrame): M = {
    this.setPipeline(dataset)
    data = new SSVData()
    data.separateData(dataset)
    this.train()
  }

  def train(): M = {  
    if (verbose) {
      print("Entering train ...\n")
      print("Unlabeled count : " + data.unlabeledData.count() + "\n")
      print("Labeled count : " + data.labeledData.count() + "\n")
    }

    if (data.unlabeledData.count > 0) {
      var numberOfIteration: Int = 0
      var countNewLabeled: Long = 0
      do {
        val model = pipelineSSV.fit(data)  
        val newData = pipelineSSV.getNewReliableData(data)
        data.rawUnlabeledData = data.rawUnlabeledData.except(newData.select("features"))
        data.unlabeledData = data.rawUnlabeledData
        data.rawLabeledData = data.rawLabeledData.unionAll(newData)
        data.labeledData = data.rawLabeledData
        countNewLabeled = newData.count
        if (verbose) {
          print("\nIteration " + (numberOfIteration+1) + "\n")
          print("New labeled count: " + countNewLabeled + "\n")
          print("Current Labeled Data count : " + data.labeledData.count() + "\n")
        }
        numberOfIteration += 1
      } while (numberOfIteration < countIterations && data.unlabeledData.count > 0 && countNewLabeled > 0)
    }
    
    pipelineSSV.constructModel(data)
  }
  
  override def copy(extra: org.apache.spark.ml.param.ParamMap):SelfLearningClassifier[FeatureType,E,M] = defaultCopy(extra)
}

def SelfLearningRFClassifier(classifier: RandomForestClassifier, data: DataFrame) = {
  val result = new SelfLearningClassifier[Vector, RandomForestClassifier, org.apache.spark.ml.classification.RandomForestClassificationModel](classifier)
  result.setPipeline(data)
  result
}


