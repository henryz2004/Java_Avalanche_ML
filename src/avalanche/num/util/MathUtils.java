package avalanche.num.util;

import java.util.List;

public class MathUtils {

    public static final double e  = 2.7183;
    public static final double pi = 3.1416;

    private MathUtils() {}

    public static double        sum(double[] list) {
        double sum = 0;
        for (double item : list) {
            sum += item;
        }
        return sum;
    }
    public static double        round(double number, int decimalPlaces) {
        double multiplier = java.lang.Math.pow(10, decimalPlaces);
        return java.lang.Math.round(number*multiplier)/multiplier;
    }
    public static double[]      multiplyCorresponding(double[] list, double[] otherList) {
        double[] resultList = new double[list.length];
        for (int i=0; i<list.length; i++) {
            resultList[i] = list[i] * otherList[i];
        }
        return resultList;
    }
    public static double[]      square(double[] list) {
        double[] resultList = new double[list.length];
        for (int i = 0; i < list.length; i++) {
            resultList[i] = list[i] * list[i];
        }
        return resultList;
    }
    public static double[]      subtractBy(List<Double> list, double number) {
        return subtractBy((double[]) (Object) list.toArray(), number);  // Safe because we KNOW that the object is actually a double[]
    }
    public static double[]      subtractBy(double[] list, double number) {
        double[] returnList = new double[list.length];
        for (int i=0; i < list.length; i++) {
            returnList[i] = list[i] - number;
        }
        return returnList;
    }
    public static void          squareInplace(double[] list) {
        for (int i=0; i < list.length; i++) {
            list[i] = list[i] * list[i];
        }
    }
    public static void          subtractByInplace(double[] list, double number) {
        for (int i=0; i < list.length; i++) {
            list[i] -= number;
        }
    }
    public static double        findAverage(List<Double> list) {
        double average = 0;
        for (double item : list) {
            average += item/list.size();
        }
        return average;
    }
    public static double        findAverageOfDimension(List<double[]> list, int dimensionIndex) {
        // Find the average of all the numbers of dimensionIndex
        // For example if the input is [1,2], [3,4], [5,6]
        // and the dimension index is 1, the value returned would be the average of 2, 4, and 6

        double average = 0;
        for (double[] item : list) {
            average += item[dimensionIndex] / list.size();
        }
        return average;
    }
}
