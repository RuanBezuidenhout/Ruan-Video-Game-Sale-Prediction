package RuanPackage

import org.apache.log4j._
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.regression.DecisionTreeRegressor
import org.apache.spark.sql._
import org.apache.spark.ml.feature.{StringIndexer, OneHotEncoder}

object RVGSP {

  case class DecisionTreeSchema(Name:String, Platform:String, Genre:String, Publisher:String, Global_Sales:Double)
  /** main function*/
  def main(args: Array[String]) {
    // Set the log level to only print errors
    Logger.getLogger("org").setLevel(Level.ERROR)

    // Creating a spark Session and importing the data
    val spark = SparkSession
      .builder
      .appName("RVGSP")
      .master("local[*]")
      .getOrCreate()

    // Importing dataset
    import spark.implicits._
    val dsRaw = spark.read
      .option("sep", ",")
      .option("header","true")
      .option("inferSchema","true")
      .csv("data/vgsales_clean.csv")
      .as[DecisionTreeSchema]

    //Converting categorical variables to numeric
    //Converting column of string values to a column of label indexes
    val indexer = new StringIndexer()
      .setInputCols(Array("Platform","Genre","Publisher"))
      .setOutputCols(Array("PlatformIndex","GenreIndex","PublisherIndex"))
      .fit(dsRaw)
    val indexed = indexer.transform(dsRaw)

    //Secondly converting column of indexes to a column of binary vectors
    val encoder = new OneHotEncoder()
      .setInputCols(Array("PlatformIndex","GenreIndex","PublisherIndex"))
      .setOutputCols(Array("PlatformVec","GenreVec","PublisherVec"))
      .fit(indexed)
    val encoded = encoder.transform(indexed)

    // Setting up vectors based on the case classes
    val assembler = new VectorAssembler().
      setInputCols(Array("PlatformVec","GenreVec","PublisherVec")).
      setOutputCol("features")
    val df = assembler.transform(encoded)
      .select("Global_Sales","features")

    // Splitting the data into training data and testing data
    val trainTest = df.randomSplit(Array(0.5, 0.5))
    val trainingDF = trainTest(0)
    val testDF = trainTest(1)

    // Creating the Decision Tree Regression model
    val dtr = new DecisionTreeRegressor()
      .setFeaturesCol("features")
      .setLabelCol("Global_Sales")

    // Training the model using our training data
    val model = dtr.fit(trainingDF)

    // Now the model tries to predict the values
    val fullPredictions = model.transform(testDF).cache()

    // Extract the predictions and the correct values
    val predictionAndLabel = fullPredictions.select("prediction", "Global_Sales").collect()

    // Print out the predicted and the actual values
    for (prediction <- predictionAndLabel) {
      println(prediction)
    }

    // Stop the session
    spark.stop()

  }
}