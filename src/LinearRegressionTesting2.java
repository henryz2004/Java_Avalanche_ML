import avalanche.data.Dataset;
import avalanche.data.regression.linear.LinearRegression;
import avalanche.num.Matrix;
import avalanche.num.util.MathUtils;

public class LinearRegressionTesting2 {

    public static void main(String[] args) {

        // Sqaure feet to cost of house
        Dataset dataset1 = new Dataset (
                Matrix.from2D( new double[][] {
                        new double[] {2296, 569900},
                        new double[] {2000, 569900},
                        new double[] {2061, 479900},
                        new double[] {2464, 580000},
                        new double[] {4878, 975000},
                        new double[] {1287, 339900},
                        new double[] {3209, 789000},
                        new double[] {1201, 399900},
                        new double[] {7000, 1650000},
                        new double[] {3284, 775000},
                        new double[] {3500, 680000}
                })
        );
        LinearRegression regressor = new LinearRegression(dataset1);

        System.out.println(regressor.regress());
        System.out.println(regressor.rSquared);
        System.out.println(regressor.equation.findY(3500));

        // We should round rSquared
        System.out.println(MathUtils.round(regressor.rSquared, 2));
    }
}
