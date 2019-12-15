package avalanche.data;

import avalanche.num.Matrix;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class Dataset {

    public List<double[]> data;
    public List<double[]> centroids;
    public List<List<double[]>> clusters;      // List (clustersReference) of lists (datapoints) of points (double[])

    public Dataset() {
        data      = new ArrayList<>();
        centroids = new ArrayList<>();
        clusters  = new ArrayList<>();
    }
    public Dataset(double[][] startData) {
        data      = new ArrayList<>(Arrays.asList(startData));
        centroids = new ArrayList<>();
        clusters  = new ArrayList<>();
    }
    public Dataset(Matrix matrix) {
        data      = new ArrayList<>(Arrays.asList(matrix.toArray()));
        centroids = new ArrayList<>();
        clusters  = new ArrayList<>();
    }

    public void addData(double[] newData) {
        data.add(newData);
    }
}
