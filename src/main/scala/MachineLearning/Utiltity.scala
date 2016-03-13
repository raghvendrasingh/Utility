package MachineLearning

import breeze.linalg.DenseMatrix
import java.io.FileNotFoundException
import java.security.InvalidParameterException
import scala.io.Source
import scala.collection.mutable.ListBuffer

/**
  * Created by raghvendra.singh on 1/13/16.
  */
object Utiltity {

  /** This method checks whether a string is null or empty
    *
    * @param str - An input string
    * @return - A boolean flag returning 1 if str is null or empty otherwise 0
    */
  def isNotDefined(str: String): Boolean = {
    str == null || str.trim.isEmpty || str.isEmpty
  }

  /** This method loads the data from a csv file
    *
    * @param fileName - CSV fileName containing the data. All the data values should be double.This method
    *                 populates the sampleData vector.
    */
  def loadData(fileName: String, delimiter: String): Vector[Vector[Double]] = {
    var sampleData = Vector[Vector[Double]]()
    val bufferedSource = io.Source.fromFile(fileName)
    var cols: Array[String] = null
    try {
      for (line <- bufferedSource.getLines) {
        cols = line.trim.split(delimiter).map(_.trim)
        val sample = cols map (x => x.toDouble)
        sampleData = sampleData :+ sample.toVector
      }
    } catch {
      case ex: NumberFormatException => println(s"Cannot convert a string ${cols} into its proper data type.")
      case ex: FileNotFoundException => println(s"The csv file ${fileName} does not exist")
      case ex: Exception => println("Unexpected execution error while executing method loadData()")
    } finally  {
      if (bufferedSource != null) bufferedSource.close()
    }
    sampleData
  }

  /** This method loads data from a CSV file into DenseMatrix. Each row of CSV is a data sample. */
  def loadDenseMatrixFromCSV(fileName: String, delimiter: String): DenseMatrix[Double] = {
    var res: DenseMatrix[Double] = null
    var arr = Array[Double]()
    try {
      var numSamples = 0
      for (line <- Source.fromFile(fileName).getLines) {
        arr = arr ++ line.split(delimiter).map (x => x.toDouble)
        if (numSamples == 0) numSamples = arr.size
      }
      res = new DenseMatrix(numSamples,arr,0)
    } catch {
      case ex: Exception => println("Unexpected execution error while executing method loadDenseMatrixFromCSV()",ex)
    }
    res
  }

  /** This method does element wise addition of two vectors and return the resultant vector
    *
    * @param a - A vector[Double]
    * @param b - A vector[Double]
    * @return -  A Vector[Double]
    */
  def addVectors(a: Vector[Double], b: Vector[Double]): Vector[Double] = {
    var vec: Vector[Double] = Vector()
    assert(a.size == b.size)
    for (i <- a.indices) {
      vec = vec :+ (a(i) + b(i))
    }
    vec
  }


  /** This method does element wise subtraction of two vectors and return the resultant vector
    *
    * @param a - A vector[Double]
    * @param b - A vector[Double]
    * @return -  A Vector[Double]
    */
  def subtractVectors(a: Vector[Double], b: Vector[Double]): Vector[Double] = {
    var vec: Vector[Double] = Vector()
    assert(a.size == b.size)
    for (i <- a.indices) {
      vec = vec :+ (a(i) - b(i))
    }
    vec
  }

  /** This method does element wise division of two vectors and return the resultant vector
    *
    * @param a - A vector[Double]
    * @param b - A vector[Double]
    * @return -  A Vector[Double]
    */
  def divideVectors(a: Vector[Double], b: Vector[Double]): Vector[Double] = {
    var vec: Vector[Double] = Vector()
    assert(a.size == b.size)
    for (i <- a.indices) {
      vec = vec :+ (a(i) / b(i))
    }
    vec
  }

  /** This method calculates the euclidean distance between two vectors.
    *
    * @param point1 - first vector.
    * @param point2 - second vector.
    * @return - Euclidean distance between two vectors.
    */
  def findDistance(point1: Vector[Double], point2: Vector[Double]): Double = {
    assert(point1.size == point2.size)
    var sum = 0.0
    try {
      for (i <- point1.indices) {
        sum = sum + math.pow(point1(i) - point2(i), 2)
      }
      sum = math.pow(sum, 0.5)
    } catch {
      case ex: Exception => println("Unexpected execution error while executing method findDistance()")
        throw ex
    }
    sum
  }

  /** This method calculates the mean of each sample in data. */
  def sampleMean(data: Vector[Vector[Double]]): Vector[Double] = {
    data map (vec => {vec.sum/vec.size})
  }

  /** This method calculates the standard deviation of each sample in data. */
  def sampleStandardDeviation(data: Vector[Vector[Double]], mean: Vector[Double]): Vector[Double] = {
    assert(data.size == mean.size)
    var stdDev = Vector[Double]()
    val numFeatures = data(0).size
    try {
      for (i <- data.indices) {
        stdDev = stdDev :+ math.sqrt(data(i).foldLeft(0.0){(x,y) => x + math.pow(y-mean(i),2)}/numFeatures)
      }
    } catch {
      case ex: Exception => println("Unexpected execution error while executing method findDistance()")
        throw ex
    }
    stdDev
  }

  /** This method calculates the mean of each feature in data. */
  def featureMean(data: Vector[Vector[Double]]): Vector[Double] = {
    val sum = data reduceLeft((x,y)=> addVectors(x,y))
    val mean = sum map (x => x/data.size)
    mean
  }

  /** This method calculates the standard deviation of each feature in data. */
  def featureStandardDeviation(data: Vector[Vector[Double]], mean: Vector[Double]): Vector[Double] = {
    assert(data(0).size == mean.size)
    var stdDev = Vector[Double]()
    val numSamples = data.size
    try {
      stdDev = data.foldLeft(Vector.fill(data(0).size)(0.0)){(p,q) => addVectors(p, subtractVectors(q,mean) map(r => r*r))} map(s => s/numSamples) map(t => math.sqrt(t))
    }catch {
      case ex: Exception => println("Unexpected execution error while executing method findDistance()")
        throw ex
    }
    stdDev
  }

  /** This method subtracts the sample mean from each element in the sample. */
  def centerDataSampleWise(data: Vector[Vector[Double]], mean: Vector[Double]): Vector[Vector[Double]] = {
    var centeredData = Vector[Vector[Double]]()
    try {
      for (i <- data.indices) centeredData = centeredData :+ (data(i) map (x => {x - mean(i)}))
    }catch {
      case ex: Exception => println("Unexpected execution error while executing method findDistance()")
        throw ex
    }
    centeredData
  }

  /** This method subtracts the feature mean from each element in the feature. */
  def centerDataFeatureWise(data: Vector[Vector[Double]], mean: Vector[Double]): Vector[Vector[Double]] = {
    var centeredData = Vector[Vector[Double]]()
    try {
      for (i <- data.indices) centeredData = centeredData :+ subtractVectors(data(i), mean)
    }catch {
      case ex: Exception => println("Unexpected execution error while executing method findDistance()")
        throw ex
    }
    centeredData
  }

  /** This method finds the minimum and maximum value inside each feature. */
  private def findMinMaxFeatureWise(data: Vector[Vector[Double]]): ListBuffer[(Double,Double)] = {
    val res = ListBuffer.fill(data(0).size)((Double.MaxValue, Double.MinValue))
    for (i <- data.indices) {
      for (j <- data(i).indices) {
        if (data(i)(j) < res(j)._1) res(j) = (data(i)(j), res(j)._2)
        if (data(i)(j) > res(j)._2) res(j) = (res(j)._1, data(i)(j))
      }
    }
    res
  }


  /** This method scales the data using either Z-score normalization or min-max scaling.
    *
    * @param data - Data to be scaled.
    * @param method - If "Z-score" then Z-score normalization. If "min-max" then min-max scaling. Z-score by default.
    * @param axis - axis used to compute the means and standard deviations along. If 0,independently
    *             standardize each feature, otherwise (if 1) standardize each sample. 0 is default.
    * @param with_mean - true by default. If true then center the data before scaling.
    * @param with_std - true by default. If true then scale the data to unit variance.
    * @return - Scaled data.
    */
  def scaleData(data: Vector[Vector[Double]], method: String = "Z-score", axis: Int = 0, with_mean: Boolean = true, with_std: Boolean = true): Vector[Vector[Double]] = {
    var scaledData = Vector[Vector[Double]]()
    var mean =  Vector[Double]()
    var stdev = Vector[Double]()
    var centeredData = data
    var min = 0.0
    var max = 0.0
    try {
      method match {
        case "Z-score" => println("Performing Z-score scaling on data.")
          if (axis == 1) {
            /** standardize each sample */
            mean = sampleMean(data)
            stdev = sampleStandardDeviation(data, mean)
            if (with_mean) {
              println("Centering the data")
              centeredData = centerDataSampleWise(data, mean)
            }
            for (i <- centeredData.indices) scaledData = scaledData :+ (centeredData(i) map (x => x/stdev(i)))
          } else if(axis == 0) {
            /** standardize each feature */
            mean = featureMean(data)
            stdev = featureStandardDeviation(data, mean)
            if(with_mean) {
              println("Centering the data")
              centeredData = centerDataFeatureWise(data, mean)
            }
            scaledData = centeredData map(x => divideVectors(x, stdev))
          } else throw new InvalidParameterException(s"$axis is invalid choice for axis. axis can be 0 or 1.")
        case "min-max" => println("Performing min-max scaling on data.")
          if (axis == 1) {
            /** standardize each sample */
            for (i <- data.indices) {
              min = data(i).min
              max = data(i).max
              scaledData = scaledData :+ (data(i) map (x => {(x-min)/(max-min)}))
            }
          } else if (axis == 0) {
            /** standardize each feature */
            val minMaxFeatureWise = findMinMaxFeatureWise(data)
            for (i <- data.indices){
              var vec = Vector[Double]()
              for (j <- data(i).indices) {
                vec = vec :+ (data(i)(j)-minMaxFeatureWise(j)._1)/(minMaxFeatureWise(j)._2 - minMaxFeatureWise(j)._1)
              }
              scaledData = scaledData :+ vec
            }
          } else throw new InvalidParameterException(s"$axis is invalid choice for axis. axis can be 0 or 1.")
        case _ => println(s"$method is invalid scaling method. Z-score and min-max are the only allowed options.")
          throw new InvalidParameterException(s"$method is invalid scaling method. Z-score and min-max are the only allowed options.")
      }
    } catch {
      case ex: InvalidParameterException => throw ex
      case ex: Exception => println("Unexpected execution error while executing method findDistance()")
        throw ex
    }
    scaledData
  }

}



















