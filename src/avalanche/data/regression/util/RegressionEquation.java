package avalanche.data.regression.util;

import avalanche.num.Matrix;
import avalanche.num.util.MathUtils;

public class RegressionEquation {

    public double B0;   // Y-intercept
    public Matrix coefficients;

    public RegressionEquation(Matrix coefficientVector, double b) {
        // coefficientVector is a COLUMN vector of coefficients for the linear equation
        coefficients = coefficientVector.transpose();
        B0           = b;
    }

    public double findY(Matrix variableMatrix) {
        // Dot multiplying coefficients and variableMatrix gives a 1x1 matrix which is our y
        // without applying the intercept
        // variableMatrix should be a row vector

        Matrix yFromCoefficients = Matrix.dotProduct(/*coefficients, variableMatrix*/variableMatrix, coefficients);     // FIXME Check on this

        return yFromCoefficients.getAt(0,0) + B0;
    }
    public double findIntercept(double[] coordinate) {

        // Map each coordinate with its corresponding coefficient, so that way
        // each coefficient has a 'variable'
        // Compute the coefficients * variables, subtract that from the y in order to find b0

        // Used for multiply and sum. Last column is the y (a.k.a. the dependent variable, the result)
        Matrix rowVector = Matrix.from1D(coordinate).exceptColumn(-1);
        double nonInterceptTerm = Matrix.dotProduct(rowVector, coefficients).getAt(0,0);

        // TODO: Remove printlines
        System.out.println("[!] RegressionEquation outs (in findIntercept()):");
        System.out.println("    rowVector:      " + rowVector.toCleanString());
        System.out.println("    coefficients:   " + coefficients.toCleanString());
        System.out.println();

        // Therefore, because y = b0 + nonInterceptTerm, b0 = y-nIT
        return coordinate[coordinate.length-1] - nonInterceptTerm;
    }

    @Override
    public String toString() {
        return toString(true);
    }
    public String toString(boolean round) {
        StringBuilder sb = new StringBuilder();
        sb.append("y = ");
        if (round) sb.append(MathUtils.round(B0, 3));
        else       sb.append(B0);
        sb.append(" ");

        int subscript = 1;

        for (double coefficient : coefficients.flatten()) {
            if (coefficient >= 0) {
                sb.append("+ ");
                if (round) sb.append(MathUtils.round(coefficient, 3));
                else       sb.append(coefficient);

            } else {
                sb.append("- ");
                if (round) sb.append(MathUtils.round(-coefficient, 3));
                else       sb.append(-coefficient);
            }
            sb.append("*B");
            sb.append(subscript);
            sb.append(" ");

            subscript++;
        }
        return sb.toString();
    }
}
