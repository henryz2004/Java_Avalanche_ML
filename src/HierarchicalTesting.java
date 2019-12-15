import avalanche.data.Dataset;
import avalanche.data.cluster.hierarchical.Hierarchical;
import avalanche.num.Matrix;

import java.util.Arrays;
import java.util.List;

public class HierarchicalTesting {

    public static void main(String[] args) {

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
        Hierarchical clusterer = new Hierarchical(dataset1);

        clusterer.fit(10,2);
        for (List<double[]> cluster : dataset1.clusters) {
            System.out.print(Arrays.deepToString(cluster.toArray()) + " ");
        }
        System.out.println();
        System.out.println(Arrays.deepToString(dataset1.centroids.toArray()));
        System.out.println(Arrays.deepToString(clusterer.dendrogram.toArray()));
    }
}
