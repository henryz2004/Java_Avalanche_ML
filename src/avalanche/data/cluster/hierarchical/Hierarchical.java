package avalanche.data.cluster.hierarchical;

import avalanche.data.Dataset;
import avalanche.data.cluster.util.ClusterUtils;
import avalanche.data.cluster.util.LinkageCriteria;
import avalanche.data.cluster.util.Metric;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class Hierarchical {
    // Objected-oriented in order to encapsulate some settings or attributes
    // Make one of this for each dataset you're analyzing

    // These are REFERENCES (except for dendrogram) to the real thing in the DATASET
    public List<double[]> dataReference;
    public List<double[]> centroidsReference;
    public List<double[]> dendrogram;
    public List<List<double[]>> clustersReference;   // Lists of clustersReference and what's in those clustersReference

    private Dataset         datasetReference;
    private Metric          metric;
    private LinkageCriteria linkageCriteria;

    public Hierarchical(Dataset dataset) {
        this(dataset, Metric.EUCLIDEAN, LinkageCriteria.COMPLETE);
    }
    public Hierarchical(Dataset dataset, Metric measurementMetric, LinkageCriteria linkCriteria) {
        dataReference      = dataset.data;
        centroidsReference = dataset.centroids;
        clustersReference  = dataset.clusters;
        dendrogram = new ArrayList<>();

        datasetReference = dataset;
        metric           = measurementMetric;
        linkageCriteria  = linkCriteria;
    }

    public void setMetric(Metric newMetric) {
        metric = newMetric;
    }
    public void setLinkageCriteria(LinkageCriteria newLinkageCriteria) {
        linkageCriteria = newLinkageCriteria;
    }

    public void fit(double maxDistance, int minClusters) {
        // TODO: Auto detect when to stop clustering

        if (maxDistance == Double.POSITIVE_INFINITY || minClusters <= 0) {
            throw new IllegalArgumentException(
                    "maxDistance must be less than positive infinity, and minClusters must be greater than 0"
            );
        }

        // Make each of our dataReference points its own little cluster
        clustersReference.clear();

        for (double[] datapoint : dataReference) {
            clustersReference.add(new ArrayList<>(Arrays.asList(datapoint)));
        }

        // Distance matrix: https://en.wikipedia.org/wiki/Distance_matrix
        List<List<Double>> distanceMatrix = calculateDistanceMatrix();

        // While 1) we still have dataReference to fit (goes with number 3)
        //       2) our max distance so far is less than the maxDistance
        //       3) the number of clustersReference is greater than the minClusters (this is implemented in mergeCluster)
        // continue to fit
        while (clustersReference.size() > minClusters && mergeCluster(distanceMatrix, maxDistance)) {
            distanceMatrix = calculateDistanceMatrix();     // Recalculate the distance matrix
        }

        ClusterUtils.calculateCentroids(datasetReference);
    }

    // Private helper methods that aren't in the util.ClusterUtils class/too specialized
    private double calculateDistanceCluster(int index, int otherIndex) {
        return calculateDistanceCluster(clustersReference.get(index), clustersReference.get(otherIndex));
    }
    private double calculateDistanceCluster(List<double[]> cluster, List<double[]> otherCluster) {

        double distance = 0;
        switch (linkageCriteria) {

            // Find the farthest distance between the clustersReference
            // Sadly this is O(n^2)
            case COMPLETE:

                for (double[] coordinate : cluster) {
                    for (double[] otherCoordinate : otherCluster) {
                        double currentDistance = ClusterUtils.calculateDistance(coordinate, otherCoordinate, metric);
                        if (currentDistance > distance) distance = currentDistance;
                    }
                }

            // The opposite of complete
            case SINGLE:
                distance = Double.POSITIVE_INFINITY;    // In order to minimize we must start from the very max

                for (double[] coordinate : cluster) {
                    for (double[] otherCoordinate : otherCluster) {
                        double currentDistance = ClusterUtils.calculateDistance(coordinate, otherCoordinate, metric);
                        if (currentDistance < distance) distance = currentDistance;
                    }
                }

            // Average
            case AVERAGE:
                int numberOfDistances = cluster.size() * otherCluster.size();

                for (double[] coordinate : cluster) {
                    for (double[] otherCoordinate : otherCluster) {
                        distance += ClusterUtils.calculateDistance(coordinate, otherCoordinate, metric)/numberOfDistances;
                    }
                }
        }

        return distance;
    }
    private List<List<Double>> calculateDistanceMatrix() {
        // At LEAST O(n^2)

        List<List<Double>> distanceMatrix = new ArrayList<>();

        for (int r=0; r<clustersReference.size(); r++) {
            distanceMatrix.add(new ArrayList<>());
            for (int c=0; c<clustersReference.size(); c++) {
                if (r==c) distanceMatrix.get(r).add(0D);
                else      distanceMatrix.get(r).add(calculateDistanceCluster(r, c));
            }
        }

        return distanceMatrix;
    }
    private boolean mergeCluster(List<List<Double>> distanceMatrix, double maxDistance) {

        // Find the closest two clustersReference
        int firstClosest  = 0;  // Row index
        int secondClosest = 0;  // Column index
        double bestDistance = Double.POSITIVE_INFINITY;

        // j=i due to the fact that the important part of the matrix is a triangle
        for (int i=0; i<distanceMatrix.size(); i++) {
            for (int j=i; j<distanceMatrix.size(); j++) {

                double distance = distanceMatrix.get(i).get(j);

                if (distance < bestDistance && distance != 0) {
                    firstClosest = i;
                    secondClosest = j;
                    bestDistance = distance;
                }

            }
        }
        System.out.println(bestDistance + " " + maxDistance);
        ClusterUtils.print2DList(distanceMatrix);

        // If the closest distance is less than the max distance, merge.
        if (bestDistance < maxDistance) {
            // Merge clustersReference first and second together in our clustersReference
            clustersReference.get(firstClosest).addAll(clustersReference.get(secondClosest));
            clustersReference.remove(secondClosest);

            // Add this merge into the dendrogram
            // first item  is the new cluster's index (also the index of the first cluster that was merged)
            // second item is the cluster that was removed (also the index of the second cluster that was merged)
            // third item  is the distance between the clustersReference
            // fourth item is the number of elements in this new cluster
            dendrogram.add(new double[] {firstClosest, secondClosest, bestDistance, clustersReference.get(firstClosest).size()});

            return true;
        }

        // Return false- we did not merge
        return false;
    }
}
