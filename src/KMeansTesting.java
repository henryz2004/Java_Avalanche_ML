import avalanche.data.Dataset;
import avalanche.data.cluster.kmeans.KMeans;
import avalanche.data.cluster.util.ClusterMode;
import avalanche.num.Matrix;

import java.util.Arrays;

public class KMeansTesting {

    public static void main(String[] args) throws Exception{

        Dataset dataset1 = new Dataset(
              Matrix.from2D( new double[][] {
                      new double[] {1, 2},
                      new double[] {5, 8},
                      new double[] {1.5, 1.8},
                      new double[] {8, 8},
                      new double[] {1, 0.6},
                      new double[] {9, 11}
              })
        );
        KMeans clusterer = new KMeans(dataset1);

        clusterer.fit(ClusterMode.HARTIGAN_WONG,2,5);

        System.out.println(Arrays.deepToString(dataset1.centroids.toArray()));
    }
}
