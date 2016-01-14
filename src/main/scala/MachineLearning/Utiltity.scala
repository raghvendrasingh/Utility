package MachineLearning

/**
  * Created by raghvendra.singh on 1/13/16.
  */
object Utiltity {
  def isNullOrEmptyOrWhiteSpace(str: String): Boolean = {
    str == null || str.trim.isEmpty || str.isEmpty
  }
}
