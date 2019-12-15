package avalanche.data.cluster.kmeans;

import avalanche.data.Dataset;
import avalanche.data.cluster.util.ClusterMode;
import avalanche.data.cluster.util.ClusterUtils;
import avalanche.data.cluster.util.Metric;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ThreadLocalRandom;

public class KMeans {
    // Object oriented K-Means. Make one for each dataset you're analyzing

    // References to the attributes in the Dataset this object is analyzing
    public List<double[]> dataReference;
    public List<double[]> centroidsReference;
    public List<List<double[]>> clustersReference;      // List (clustersReference) of lists (datapoints) of points (double[])

    private Dataset datasetReference;
    private Metric metric;

    // Public constructors
    public KMeans(Dataset dataset) {
        this(dataset, Metric.EUCLIDEAN);
    }
    public KMeans(Dataset dataset, Metric measurementMetric) {
        dataReference      = dataset.data;
        centroidsReference = dataset.centroids;
        clustersReference  = dataset.clusters;

        datasetReference = dataset;
        metric = measurementMetric;
    }

    public void setMetric(Metric newMetric) {
        metric = newMetric;
    }

    // Supports Lloyd and Hartigan-Wong
    public void fit(int numClusters) {
        fit(ClusterMode.HARTIGAN_WONG, numClusters, 2);   // Default is hartigan wong
    }
    public void fit(ClusterMode clusterMode, int numClusters, int iterations) {

        if (dataReference == null) throw new IllegalArgumentException("Uninitialized datasetReference, cannot fit");

        if (clusterMode == ClusterMode.HARTIGAN_WONG) {

            // Hartigan Wong K-Means

            // First empty out all the clustersReference
            clustersReference.clear();
            for (int i=0; i<numClusters; i++) {
                clustersReference.add(new ArrayList<>());
            }

            // The assign the datapoints randomly
            for (double[] datapoint : dataReference) {
                int clusterIndex = ThreadLocalRandom.current().nextInt(clustersReference.size());
                clustersReference.get(clusterIndex).add(datapoint);
            }

            ClusterUtils.calculateCentroids(datasetReference);

            for (int i=0; i<iterations; i++) {

                for (double[] datapoint : dataReference) {

                    int nearestCentroidIndex = ClusterUtils.getCentroidClosest(datasetReference, datapoint, metric);
                    int originalCentroidIndex = 0;

                    // We also need the original centroid index
                    for (int j = 0; j< centroidsReference.size(); j++) {
                        if (clustersReference.get(j).contains(datapoint)) {
                            originalCentroidIndex = j;
                            break;
                        }
                    }

                    clustersReference.get(nearestCentroidIndex).add(datapoint);
                    clustersReference.get(originalCentroidIndex).remove(datapoint);

                    if (originalCentroidIndex != nearestCentroidIndex) {
                        ClusterUtils.calculateCentroids(datasetReference);
                    }

                }
            }

        } else {

            // Lloyd K-Means
            ClusterUtils.randomCentroids(datasetReference, numClusters);
            ClusterUtils.calculateClusters(datasetReference, metric);        // Make sure to update our clustersReference before we start to refine them

            for (int i=0; i<iterations; i++) {

                centroidsReference.clear();      // Fresh start

                for (List<double[]> cluster : clustersReference) {
                    if (cluster.size() <= 0) continue;      // Skip over the empty clustersReference

                    // The centroid after all the averaging
                    double[] averagedCentroid = new double[cluster.get(0).length];

                    for (int dimension = 0; dimension < cluster.get(0).length; dimension++) {
                        for (double[] datapoint : cluster) {
                            averagedCentroid[dimension] += datapoint[dimension];
                        }
                    }

                    // Now average each item
                    for (int j = 0; j < averagedCentroid.length; j++) {
                        averagedCentroid[j] = averagedCentroid[j] / cluster.size();
                    }

                    centroidsReference.add(averagedCentroid);
                }

                ClusterUtils.calculateClusters(datasetReference, metric);        // Centroids have changed, and so will the clustersReference
            }

        }
    }
}
