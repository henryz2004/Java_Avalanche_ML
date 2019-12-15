import avalanche.data.Dataset;
import avalanche.data.regression.linear.LinearRegression;
import avalanche.num.Matrix;
import avalanche.num.util.MathUtils;

public class LinearRegressionTesting {

    public static void main(String[] args) {

        Dataset dataset1 = new Dataset (
                Matrix.from2D( new double[][] {
                        new double[] {1, 2},
                        new double[] {2, 4},
                        new double[] {3, 5},
                        new double[] {4, 4},
                        new double[] {5, 5}
                })
        );
        LinearRegression regressor = new LinearRegression(dataset1);

        System.out.println(regressor.regress());
        System.out.println(regressor.rSquared);

        // We should round rSquared
        System.out.println(MathUtils.round(regressor.rSquared, 2));
    }
}
