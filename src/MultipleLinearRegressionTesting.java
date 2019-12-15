import avalanche.data.CorrelationDataset;
import avalanche.data.Dataset;
import avalanche.data.regression.linear.MultipleLinearRegression;
import avalanche.num.Matrix;

import java.util.Arrays;
import java.util.Collections;

import static avalanche.util.ArrayUtils.a;
import static avalanche.util.ArrayUtils.aL;

public class MultipleLinearRegressionTesting {

    public static void main(String[] args) {

        // Equation is a 'noised' version of y = 5 + 3x + 4x
        // FIXME: R square is over 1?
        Dataset backingDataset = new Dataset(
                Matrix.from2D( new double[][] {
                        a(1D, 1, 12),
                        a(1D, 2, 16),
                        a(2D, 1, 15),
                        a(2D, 2, 19),
                        a(2D, 3, 23),
                        a(3D, 2, 22),
                        a(3D, 3, 26)/*,
                        a(3, 4, 30),
                        a(4, 5, 37),
                        a(5, 6, 44),
                        a(6, 7, 51)*/
                })
        );

        CorrelationDataset dataset1 = new CorrelationDataset(backingDataset);
        dataset1.labelIndependents(aL("X1", "X2"));
        dataset1.labelDependents(Collections.singletonList("Y"));

        System.out.println(Arrays.deepToString(backingDataset.data.toArray()));
        System.out.println(Arrays.deepToString(dataset1.independentData.toArray()));

        MultipleLinearRegression regressor = new MultipleLinearRegression(dataset1);

        regressor.regress();
        regressor.printResults();
    }
}
