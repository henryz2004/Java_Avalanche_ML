package avalanche.data.regression.util;

public class LinearEquation {
    // Class that can store linear equations of the form y = mx + b

    public double slope;
    public double yIntercept;

    public LinearEquation(double m, double b) {
        slope      = m;
        yIntercept = b;
    }

    public double findX(double y) {
        return (y - yIntercept)/slope;
    }
    public double findY(double x) {
        return slope*x + yIntercept;
    }

    @Override
    public String toString() {
        if (yIntercept > 0) {
            return "y = " + slope + "x + " + yIntercept;
        } else if (yIntercept < 0) {
            return "y = " + slope + "x - " + -yIntercept;
        } else {
            return "y = " + slope + "x";
        }
    }
}
