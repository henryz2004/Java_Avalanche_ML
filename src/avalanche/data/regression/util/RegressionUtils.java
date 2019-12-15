package avalanche.data.regression.util;

import avalanche.data.regression.linear.LinearRegression;
import avalanche.data.regression.linear.MultipleLinearRegression;
import avalanche.num.Matrix;
import avalanche.num.util.MathUtils;
import avalanche.util.ArrayUtils;

import java.util.List;

public class RegressionUtils {
    // Helper class with plenty of helper methods
    // Sadly, almost all of these methods are specialized
    // towards doubles

    private RegressionUtils() {}

    public static double         calculateRSquared(LinearRegression regression, double[] meanDifferences, double meanY) {

        // R squared is E((predictedYs - averageYs)^2)/E(meanDifferences^2)
        // meanDifferences is the difference each point is from the mean
        double[] predictionResiduals = new double[regression.dataReference.size()];

        for (int i=0; i<predictionResiduals.length; i++) {
            predictionResiduals[i] = regression.equation.findY(regression.dataReference.get(i)[0]) - meanY;
        }

        // E((predictedYs - averageYs)^2)/E(meanDifferences^2)
        return MathUtils.sum(MathUtils.square(predictionResiduals))/ MathUtils.sum(MathUtils.square(meanDifferences));
    }
    public static double         calculateRSquared(MultipleLinearRegression regression, double[] meanDifferences) {

        double[] residualSumOfSquares = new double[regression.datasetReference.dependentData.size()];   // (Actual - predicted)^2

        for (int i=0; i<residualSumOfSquares.length; i++) {

            // Make a row vector containing the variables (independent data) in order to pass the variables into findY()
            double[] independentVariables = regression.datasetReference.independentData.get(i);
            Matrix rowVariableMatrix      = Matrix.from1D(independentVariables);

            double actual       = regression.datasetReference.dependentData.get(i)[0];
            double predicted    = regression.equation.findY(rowVariableMatrix);

            residualSumOfSquares[i]       =  (actual - predicted);
        }

        // 1 - residualSumOfSquares/totalSumOfSquares
        // rSS and tSS aren't summed/squared yet, therefore do sum(square(...))
        return 1 - MathUtils.sum(MathUtils.square(residualSumOfSquares))/MathUtils.sum(MathUtils.square(meanDifferences));
    }
    public static double         findStandardDeviation(List<Double> list) {
        return findStandardDeviation(list, false);      // Assume the input is the whole population
    }
    public static double         findStandardDeviation(List<Double> list, boolean sample) {
        if (!sample) {
            // SQRT((1/n) * E(xi - mean)^2)
            return Math.sqrt(
                    (1 / list.size())
                            * MathUtils.sum(
                            MathUtils.square(
                                    MathUtils.subtractBy(
                                            list, MathUtils.findAverage(list)
                                    )
                            )
                    )
            );
        } else {
            // SQRT((1/(n-1)) * E(xi - mean)^2)
            return Math.sqrt(
                    (1 / (list.size()-1))
                            * MathUtils.sum(
                            MathUtils.square(
                                    MathUtils.subtractBy(
                                            list, MathUtils.findAverage(list)
                                    )
                            )
                    )
            );
        }
    }
    public static double         findStandardDeviationOfDimension(List<double[]> list, int dimensionIndex) {
        return findStandardDeviationOfDimension(list, dimensionIndex, false);
    }
    public static double         findStandardDeviationOfDimension(List<double[]> list, int dimensionIndex, boolean sample) {
        return findStandardDeviation(ArrayUtils.getColumnAsList(list, dimensionIndex), sample);
    }
    // TODO: Box and whisker statistics
}
