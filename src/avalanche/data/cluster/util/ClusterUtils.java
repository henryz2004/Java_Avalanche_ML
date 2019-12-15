package avalanche.data.cluster.util;

import avalanche.data.Dataset;

import java.math.RoundingMode;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ThreadLocalRandom;

public class ClusterUtils {

    // Utility class therefore cannot be instantiated!!!!!!!
    private ClusterUtils() {}

    // Utility methods
    // Only ones that are useful for many purposes and not too localized are moved into this class
    /**
     * Generates random centroidsReference for the given kMeans object
     *
     * @param dataset       This is the dataset the method is generating centroidsReference for
     * @param numCentroids  This is the number of centroidsReference to be randomly generated
     */
    public static void randomCentroids(Dataset dataset, int numCentroids) {

        dataset.centroids.clear();              // Reset the dataset's centroidsReference
        int dimensions = dataset.data.get(0).length;

        // Generate random numbers
        for (int i=0; i<numCentroids; i++) {
            double[] centroid = new double[dimensions];
            for (int j=0; j<centroid.length; j++) {

                // The number at that dimension must be within the domain of the coordinates
                // Find the minimum and maximum of the dimension in dataReference
                double min = dataset.data.get(0)[j];
                double max = min;

                for (double[] data : dataset.data) {
                    double number = data[j];

                    if (number < min)       min = number;
                    else if (number > max)  max = number;
                }

                centroid[j] = ThreadLocalRandom.current().nextDouble(min, max);
            }
            dataset.centroids.add(centroid);
        }
    }
    public static void calculateClusters(Dataset dataset, Metric metric) {

        dataset.clusters.clear();   // Fresh start

        // Add # of centroidsReference empty clustersReference because that's how many clustersReference there'll be
        for (int i=0; i<dataset.centroids.size(); i++) {
            dataset.clusters.add(new ArrayList<>());
        }

        for (double[] datapoint : dataset.data) {
            int centroidIndex = getCentroidClosest(dataset, datapoint, metric);
            dataset.clusters.get(centroidIndex).add(datapoint);
        }
    }
    public static void calculateCentroids(Dataset dataset) {

        dataset.centroids.clear();

        // Simply the average of all the points in a cluster
        for (List<double[]> cluster : dataset.clusters) {
            if (cluster.size() <= 0) continue;

            double[] centroid = new double[cluster.get(0).length];

            for (int dim=0; dim<centroid.length; dim++) {
                for (double[] point : cluster) {
                    centroid[dim] += point[dim];
                }
            }

            for (int i=0; i<centroid.length; i++) {
                centroid[i] = centroid[i]/cluster.size();
            }

            dataset.centroids.add(centroid);
        }

    }
    public static int getCentroidClosest(Dataset dataset, double[] datapoint, Metric metric) {
        // Returns the index of the centroid it belongs to
        // TODO: Use Metric enums when calculating closest centroid

        int centroidIndex    = 0;
        double centroidDist  = Double.POSITIVE_INFINITY;

        for (int index=0; index < dataset.centroids.size(); index++) {
            double[] centroid = dataset.centroids.get(index);

            // Calculate the distance between datapoint and centroid
            double distance = calculateDistance(datapoint, centroid, metric);

            if (distance < centroidDist) {
                centroidDist = distance;
                centroidIndex = index;
            }
        }

        return centroidIndex;
    }
    public static double calculateDistance(double[] point, double[] otherPoint, Metric metric) {

        double distance = 0;
        switch (metric) {

            // Sum corresponding coordinates, square them and finally square root
            case EUCLIDEAN:

                for (int coordinateIndex=0; coordinateIndex < point.length; coordinateIndex++) {
                    double correspondingDifference = point[coordinateIndex] - otherPoint[coordinateIndex];
                    distance += correspondingDifference * correspondingDifference;
                }
                distance = Math.sqrt(distance);
                break;

            // Same as euclidean but no square rooting
            case SQUARED_EUCLIDEAN:

                for (int coordinateIndex=0; coordinateIndex < point.length; coordinateIndex++) {
                    double correspondingDifference = point[coordinateIndex] - otherPoint[coordinateIndex];
                    distance += correspondingDifference * correspondingDifference;
                }
                break;

            // Same as euclidean except with no squaring and square rooting
            case MANHATTAN:

                for (int coordinateIndex=0; coordinateIndex < point.length; coordinateIndex++) {
                    double correspondingDifference = Math.abs(point[coordinateIndex] - otherPoint[coordinateIndex]);
                    distance += correspondingDifference;
                }
                break;
        }

        return distance;
    }
    public static void print2DList(List<List<Double>> list2D) {
        StringBuilder output = new StringBuilder();
        DecimalFormat df     = new DecimalFormat("###.#");
        df.setRoundingMode(RoundingMode.CEILING);
        for (List<Double> list : list2D) {
            output.append("[");
            for (int index=0; index<list.size(); index++) {
                output.append(String.format("%6s", df.format(list.get(index))));
                if (index < list.size() - 1) {
                    output.append(", ");
                }
            }
            output.append("]"+System.lineSeparator());
        }
        System.out.println(output.toString());
    }
}
