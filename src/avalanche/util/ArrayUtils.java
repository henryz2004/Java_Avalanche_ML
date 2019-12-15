package avalanche.util;

import avalanche.util.annotations.Primitive;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Ease of access utilities, such as easy array instantiation, etc.
 */
@Primitive
public class ArrayUtils {

    private ArrayUtils() {}

    /**
     * Used for quickly making arrays without the long new ...[] {};
     * a  = array
     * aL = array -> List
     *
     * I swear, generics are BROKEN
     */
    public static double[] a(double... elements) {
        return elements;
    }
    public static int[]    a(int... elements) {
        return elements;
    }
    public static String[] a(String... elements) {
        return elements;
    }
    public static List<Double>  aL(Double... elements) {
        return Arrays.asList(elements);
    }
    public static List<Integer> aL(Integer... elements) {
        return Arrays.asList(elements);
    }
    public static List<String>  aL(String... elements) {
        return Arrays.asList(elements);
    }
    public static double[]       getColumn(List<double[]> list, int columnIndex) {
        double[] column = new double[list.size()];
        int index = 0;
        for (double[] row : list) {
            column[index] = row[columnIndex];
            index++;
        }
        return column;
    }
    public static double[][]     getColumns(List<double[]> list, int from, int to) {
        // From is inclusive, to is exclusive
        double[][] columns = new double[list.size()][to-from];
        int index = 0;
        for (double[] row : list) {
            int columnInsertionIndex = 0;
            for (int c=from; c < to; c++) {
                columns[index][columnInsertionIndex] = row[c];
                columnInsertionIndex++;
            }
            index++;
        }
        return columns;
    }
    public static List<Double>   getColumnAsList(List<double[]> list, int columnIndex) {
        return Arrays.asList((Double[]) (Object) getColumn(list, columnIndex)); // Safe
    }
    public static List<double[]> getColumnsAsList(List<double[]> list, int from, int to) {
        return Arrays.asList(getColumns(list, from, to));
    }
    public static double[]       concatenate(double[] array, double[] otherArray) {
        double[] result = Arrays.copyOf(array, array.length + otherArray.length);
        System.arraycopy(array, 0, result, array.length, otherArray.length);
        return result;
    }
    public static <E> E[]        concatenate(E[] array, E[] otherArray) {
        E[] result = Arrays.copyOf(array, array.length + otherArray.length);
        System.arraycopy(array, 0, result, array.length, otherArray.length);
        return result;
    }
    public static <E> List<E>    concatenateList(E[] array, E[] otherArray) {
        List<E> newList = new ArrayList<>();
        newList.addAll(Arrays.asList(array));
        newList.addAll(Arrays.asList(otherArray));
        return newList;
    }
    public static <E> List<E>    concatenateList(E[]... arrays) {
        List<E> newList = new ArrayList<>();
        for (E[] array : arrays) {
            newList.addAll(Arrays.asList(array));
        }
        return newList;
    }
}
