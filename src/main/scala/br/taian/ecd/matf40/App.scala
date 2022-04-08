package br.taian.ecd.matf40

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature._
import org.apache.spark.ml.tuning.{
  CrossValidator,
  CrossValidatorModel,
  ParamGridBuilder
}
import org.apache.spark.sql.SparkSession

object App {

  def main(args: Array[String]): Unit = {

    val spark = SparkSession.builder.master("local[*]").getOrCreate

    val table = spark.read
      .option("header", value = true)
      .option("inferSchema", value = true)
      .csv("src/main/resources/games.csv")
      .persist()

    table.printSchema
    table.show

    val seed = 11011990

    val Array(dataTrain, dataTest) = table.randomSplit(Array(0.8, 0.2), seed)

    val columns = table.columns.filter(
      !Array("gameId", "creationTime", "seasonId", "winner").contains(_)
    )

    val assembler = new VectorAssembler()
      .setInputCols(columns)
      .setOutputCol("features")

    val indexer = new StringIndexer()
      .setInputCol("winner")
      .setOutputCol("label")

    val randomForestClassifier = new RandomForestClassifier()
      .setImpurity("gini")
      .setMaxDepth(3)
      .setNumTrees(20)
      .setFeatureSubsetStrategy("auto")

    val stages = Array(assembler, indexer, randomForestClassifier)

    val pipeline = new Pipeline().setStages(stages)

    val evaluator = new BinaryClassificationEvaluator()
      .setLabelCol("label")
      .setMetricName("areaUnderROC")

    val paramGrid = new ParamGridBuilder()
      .addGrid(randomForestClassifier.maxBins, Array(25, 28, 31))
      .addGrid(randomForestClassifier.maxDepth, Array(4, 6, 8))
      .addGrid(randomForestClassifier.impurity, Array("entropy", "gini"))
      .build()

    val cv = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(5)

    val cvModel: CrossValidatorModel = cv.fit(dataTrain)

    val cvPredictionDf = cvModel.transform(dataTest)
    cvPredictionDf.show

    cvModel.write.overwrite().save("src/main/resources/model/lol")

    //    carregar o modelo do diretorio
    //    val cvModelLoaded = CrossValidatorModel.load("src/main/resources/model/lol")

  }

}
