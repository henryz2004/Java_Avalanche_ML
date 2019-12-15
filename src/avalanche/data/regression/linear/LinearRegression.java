package avalanche.data.regression.linear;

import avalanche.data.Dataset;
import avalanche.data.regression.util.LinearEquation;
import avalanche.data.regression.util.RegressionModel;
import avalanche.data.regression.util.RegressionUtils;
import avalanche.num.util.MathUtils;

import java.util.List;

public class LinearRegression {
    // Object oriented linear regression, similar concept to
    // KMeans and Hierarchical and that is make one of this
    // for each dataset you're analyzing

    public List<double[]> dataReference;
    public LinearEquation equation;
    public double rSquared;                 // R-Squared of the equation

    public LinearRegression(Dataset dataset) {
        dataReference = dataset.data;
        equation      = new LinearEquation(1, 0);   // y = x is our starting equation
        rSquared      = 0;
    }

    public LinearEquation fit() {
        return regress(RegressionModel.LEAST_SQUARES);
    }
    public LinearEquation fit(RegressionModel regressionModel) {
        return regress(regressionModel);
    }

    public LinearEquation regress() {
        return regress(RegressionModel.LEAST_SQUARES);
    }
    public LinearEquation regress(RegressionModel regressionModel) {

        if (regressionModel == RegressionModel.LEAST_SQUARES) {

            // Find the mean x value and the mean y value
            double meanX = MathUtils.findAverageOfDimension(dataReference, 0);  // X is 0th dimension (Index)
            double meanY = MathUtils.findAverageOfDimension(dataReference, 1);

            // Get a list of the differences between the x values and x average, and y and y average
            double[] meanDifferencesX = new double[dataReference.size()];
            double[] meanDifferencesY = new double[dataReference.size()];

            for (int datapointIndex = 0; datapointIndex < dataReference.size(); datapointIndex++) {
                meanDifferencesX[datapointIndex] = dataReference.get(datapointIndex)[0] - meanX;    // Subtract x value by mean
                meanDifferencesY[datapointIndex] = dataReference.get(datapointIndex)[1] - meanY;
            }

            // E((x-meanX)(y-meanY))/E((x-meanX)^2)
            // Slope of the line
            double m = MathUtils.sum(MathUtils.multiplyCorresponding(meanDifferencesX, meanDifferencesY))
                    / MathUtils.sum(MathUtils.square(meanDifferencesX));

            // We know that the linear equation has to go through the point (meanX, meanY)
            // and the slope is m, so we have meanY = m(meanX) + b, so b = meanY - m(meanX)
            double b = meanY - m * meanX;

            equation.slope = m;
            equation.yIntercept = b;

            rSquared = RegressionUtils.calculateRSquared(this, meanDifferencesY, meanY);
        }

        return equation;        // Return the equation for easy access
    }
}
