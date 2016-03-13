package MachineLearning

import scala.collection.mutable
import scala.util.Random
import java.io.FileNotFoundException
import java.security.InvalidParameterException

import scala.collection.mutable.ListBuffer
import scala.util.control.Breaks

/**
  * Created by raghvendra.singh on 1/13/16.
  */
object Utility {
  

  /** This function returns a list of only those words which do not contain any element present in punctuationList or any digit
    *
    * @param wordList - A list of words.
    * @return - A list of words without any any element present in punctuationList or any digit.
    */
  def getValidWordList(wordList: List[String]): List[String] = {
    var result = List[String]()
    val loop = new Breaks
    try {
      for (word <- wordList) {
        var flag = false
        loop.breakable {
          for (char <- word) {
            if (punctuationList.contains(char) || char.isDigit) {
              flag = true
              loop.break()
            }
          }
        }
        if (!flag) result = result :+ word
      }
      result
    } catch {
      case ex: Exception => {
        println("Unexpected execution error while executing method getValidWordList()", ex)
        throw ex
      }
    }
  }

  /** This method generates a List of random doubles in a range
    *
    * @param min - min value in the range.
    * @param max - max value in the range.
    * @param n - Number of random doubles needed.
    * @return - Return a List of n random doubles.
    */
  def getRandomDoublesInRange(min: Double, max: Double, n: Int): List[Double] = {
    var randomDoubles = ListBuffer[Double]()
    for (i <- 1 to n) randomDoubles = randomDoubles :+ min + (max - min) * Random.nextDouble()
    randomDoubles.toList
  }

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


  /** This method calculates the mean of each sample and returns a Vector of all the sample means
    *
    * @param data - A Vector[Vector[Double] ] where each internal Vector is a data sample.
    * @return - Returns a Vector of means for each data sample.
    */
  def sampleMean(data: Vector[Vector[Double]]): Vector[Double] = {
    data map (vec => {vec.sum/vec.size})
  }

  /** This method calculates the standard deviation of each data sample and returns a Vector of all the sample standard deviation.
    *
    * @param data - A Vector[Vector[Double] ] where each internal Vector is a data sample.
    * @param mean - A Vector of means for each data sample.
    * @return - Returns a Vector of all the sample standard deviation.
    */
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

  /** This method calculates feature wise mean and returns a Vector of all the feature means.
    *
    * @param data - A Vector[Vector[Double] ] where each internal Vector is a data sample.
    * @return - Returns a Vector of feature wise mean.
    */
  def featureMean(data: Vector[Vector[Double]]): Vector[Double] = {
    val sum = data reduceLeft((x,y)=> addVectors(x,y))
    val mean = sum map (x => x/data.size)
    mean
  }

  /** This method calculates the feature wise standard deviation and returns a Vector of all the feature standard deviation.
    *
    * @param data - A Vector[Vector[Double] ] where each internal Vector is a data sample.
    * @param mean - A Vector of means for each data sample.
    * @return - Returns a Vector of all the sample standard deviation.
    */
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

  /** This method does element wise subtraction of two lists and returns the resultant list
    *
    * @param a - A List[Double]
    * @param b - A List[Double]
    * @return -  A List[Double]
    */
  def subtractLists(a: List[Double], b: List[Double]): List[Double] = {
    var result = ListBuffer[Double]()
    for (i <- a.indices) result = result :+ (a(i) - b(i))
    result.toList
  }

  /** This method multiplies a scalar to each element of the list and returns the resultant list
    *
    * @param lis - A List[Double]
    * @param const - A scalar
    * @return -  A List[Double]
    */
  def multiplyScalarToList(lis: List[Double], const: Double): List[Double] = {
    var result = ListBuffer[Double]()
    for (i <- lis.indices) {
      result = result :+ lis(i) * const
    }
    result.toList
  }

  /** This method does element wise addition of two lists and return the resultant list
    *
    * @param a - A List[Double]
    * @param b - A List[Double]
    * @return -  A List[Double]
    */
  def addLists(a: List[Double], b: List[Double]): List[Double] = {
    assert(a.size == b.size)
    var result = ListBuffer[Double]()
    for (i <- a.indices) result = result :+ (a(i) + b(i))
    result.toList
  }


  /** This method subtracts the mean value of each sample from each element of the corresponding sample
    *
    * @param data - A vector of all samples.
    * @param mean - A vector which contains mean of each sample in the data.
    * @return - centered data.
    */
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

  /** This method subtracts the mean value of each feature from each element of the corresponding feature
    *
    * @param data - A vector of all samples.
    * @param mean - A vector which contains mean of each feature in the data.
    * @return - centered data.
    */
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

  /** This method calculates minimum and maximum value inside each feature in the data
    *
    * @param data - A vector of all samples.
    * @return - A list buffer of tuple containing minimum and maximum value inside each feature.
    */
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

  /** This method calculates the dot product of two lists
    *
    * @param a - first list
    * @param b - second list
    * @return - returns the scalar value which is the dot product of two lists.
    */
  def dotProductLists(a: List[Double], b: List[Double]): Double = {
    assert(a.size == b.size)
    var result = 0.0
    for (i <- a.indices) result = result + a(i) * b(i)
    result
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



















