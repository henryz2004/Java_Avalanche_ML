package avalanche.data.regression.linear;

import avalanche.data.CorrelationDataset;
import avalanche.data.regression.util.RegressionEquation;
import avalanche.data.regression.util.RegressionModel;
import avalanche.data.regression.util.RegressionUtils;
import avalanche.num.Matrix;
import avalanche.num.util.MathUtils;
import avalanche.util.MapUtils;
import avalanche.util.StringUtils;

import java.util.List;
import java.util.Map;

public class MultipleLinearRegression {

    public RegressionEquation equation;
    public double rSquared;
    public double[] standardErrors;     // Standard errors of each independent variable
    public double[] degreesOfFreedom;
    public double[] averageResidual;
    public double[] tValues;
    public double[] pValues;

    public CorrelationDataset datasetReference;
    private long timeElapsed;

    public MultipleLinearRegression(CorrelationDataset dataset) {
        equation  = new RegressionEquation(
                Matrix.fillEmpty(
                        1,
                        dataset.independentData.get(0).length,     // FIXME: Error handling in case length of data is 0
                        1
                ),
                0
        );
        rSquared        = 0;
        //standardErrors  = new double[];

        datasetReference = dataset;
        timeElapsed = 0;
    }

    public RegressionEquation fit() {
        return regress(RegressionModel.LEAST_SQUARES);
    }
    public RegressionEquation fit(RegressionModel regressionModel) {
        return regress(regressionModel);
    }

    public RegressionEquation regress() {
        return regress(RegressionModel.LEAST_SQUARES);
    }
    public RegressionEquation regress(RegressionModel regressionModel) {
        // https://stats.stackexchange.com/questions/46151/how-to-derive-the-least-square-estimator-for-multiple-linear-regression

        long startTime = System.nanoTime();

        if (regressionModel == RegressionModel.LEAST_SQUARES) {

            Matrix xMatrix = Matrix.from2D(datasetReference.independentData);   // Construct a matrix based off of independent data
            Matrix yVector = Matrix.from2D(datasetReference.dependentData);     // Same as xMatrix

            Matrix xTranspose = xMatrix.transpose();                            // Just to save time

            // b=((X′X)^−1)X′y
            // Column vector
            // WTF?! TODO: FIXME!!!!!
            // Maybe QR Decomposition or SVG
            Matrix coefficients = xTranspose.dotProduct(xMatrix).calculateInverse().dotProduct(xTranspose).dotProduct(yVector);     // FIXME: Why is this even SUPPOSED to work

            System.out.println("[!] MULTIPLE LINEAR REGRESSION DEBUG OUTS");
            System.out.println("    xMat    " + xMatrix);
            System.out.println("    xTran   " + xTranspose);
            System.out.println("    yVect   " + yVector);
            System.out.println("    inverse " + xTranspose.dotProduct(xMatrix).calculateInverse().toString());
            System.out.println();

            // The (hyper)-plane should go through the mean of all the points, am I right?
            // TODO: Confirm
            List<double[]> data = datasetReference.internalDataset.data;

            // FIXME: In case the data is empty
            double[] meanPoint = new double[data.get(0).length];

            for (int columnIndex=0; columnIndex<meanPoint.length; columnIndex++) {
                meanPoint[columnIndex] = MathUtils.findAverageOfDimension(data, columnIndex);
            }

             // Update coefficients of equation
            equation.coefficients = coefficients;

            // Plug in the mean point in order to find the intercept
            equation.B0 = equation.findIntercept(meanPoint);

            // Statistical computations
            // Calculate mean differences for y and mean y
            double   meanY              = MathUtils.findAverageOfDimension(datasetReference.dependentData, 0);
            double[] meanDifferencesY   = new double[data.size()];  // yi - ybar

            // FIXME: R square
            for (int datapointIndex = 0; datapointIndex < data.size(); datapointIndex++) {
                meanDifferencesY[datapointIndex] = datasetReference.dependentData.get(datapointIndex)[0] - meanY;
            }

            // TODO: R Square predicted, adjusted, t value, p value yadda yadda yadda
            rSquared = RegressionUtils.calculateRSquared(this, meanDifferencesY);
        }

        timeElapsed = Math.abs(System.nanoTime() - startTime);

        return equation;
    }

    /**
     * Prints the regression results in a neat and orderly manner
     *
     * Example output:
     *
     *   MULTIPLE LINEAR REGRESSION OUTPUT
     *
     *   Summary
     *    |
     *    | Regression took:                   20 milliseconds
     *    | Dataset size:                      200 points, 3D
     *    | Resulting regression equation:     y = 4.0 + 0.6*B1 + 0.275*B2
     *    | Raw equation (not rounded):        y = 4.0 + 0.6000000001*B1 + 0.274444444*B2
     *    | R Square:                          0.97
     *    | R Square (Adjusted):               0.83
     *    | R Square (Predicted):              0.69
     *    |_
     *
     *   Dependent Variable | Height
     *    |
     *    | Standard Deviation:                2.3
     *    | Residuals (Observed - Predicted):  Min: -2.39520
     *    |                                    1Q:  -0.58863
     *    |                                    Med:  0.19137
     *    |                                    3Q:   1.27642
     *    |                                    Max:  2.02364
     *    |_
     *
     *   Independent Variables/Intercept
     *    |
     *    |---------------------------------------------------------------------------------------------------------------------
     *    | Label         Coefficient     Standard Error      Degrees of Freedom      Avg. Residual       t value      p value  |
     *    |---------------------------------------------------------------------------------------------------------------------|
     *    | Intercept             4.0             0.658                        ?                 ?             ?            ?   |
     *    | Weight                0.6                 ?                        ?       even exist?             ?            ?   |
     *    | Age                 0.275                 ?                        ?       even exist?             ?            ?   |
     *    |_____________________________________________________________________________________________________________________|
     *
     *    Add Standardized Coefficient?
     */
    public void printResults() {

        StringBuilder results = new StringBuilder();

        // LinkedHashMap to store all the data being printed (except for Independent Variables)
        // Of course, it's possible to do Independent Variables, but it'll be dirty so why bother
        // Map<String, ...> are the headers and their contents
        //           , Map<String, String[]> are the sub-headers and their values
        Map<String, Map<String, String[]>> outputTree = MapUtils.<String, Map<String, String[]>>builder()
                .put(
                        "Summary", MapUtils.<String, String[]>builder()
                                .put(   // timeElapsed is in nanoseconds
                                        "Regression took:",
                                        new String[]{
                                                MathUtils.round(timeElapsed / 1000000D, 3) + " milliseconds"
                                        }
                                )
                                .put(
                                        "Dataset size:",
                                        new String[] {
                                                datasetReference.independentData.size()
                                                        + " points, "
                                                        + datasetReference.internalDataset.data.get(0).length
                                                        + "D"
                                        }
                                )
                                .put(
                                        "Resulting regression equation:",
                                        new String[] {
                                                equation.toString()
                                        }
                                )
                                .put(
                                        "Raw equation (not rounded):",
                                        new String[] {
                                                equation.toString(false)
                                        }
                                )
                                .put(
                                        "R square:",
                                        new String[] {
                                                Double.toString(MathUtils.round(rSquared, 3))
                                        }
                                )
                                .put(
                                        "R square (Adjusted):",
                                        new String[] {}             // TODO: R square adjusted
                                )
                                .put(
                                        "R square (Predicted):",
                                        new String[] {}             // TODO: R square predicted
                                )
                                .build()
                )
                .put(
                        "Dependent Variable | " + datasetReference.getIndependentLabel(0), MapUtils.<String, String[]> builder()
                                .put(
                                        "Standard Deviation:",
                                        new String[] {}             // TODO: Standard Deviation
                                )
                                .put(
                                        "Residuals (Observed - Predicted):",
                                        new String[] {"nil","nil","nil","nil","nil"}             // TODO: Residuals
                                )
                                .build()
                )
                .build();

        results.append("MULTIPLE LINEAR REGRESSION OUTPUT");
        results.append(System.lineSeparator());
        results.append(System.lineSeparator());

        // Append the output tree for each section
        for (Map.Entry<String, Map<String, String[]>> section : outputTree.entrySet()) {

            // Append the section header
            results.append(section.getKey());
            results.append(System.lineSeparator());
            results.append(" | ");
            results.append(System.lineSeparator());

            // Append the sub-headers and then the values
            for (Map.Entry<String, String[]> subSection : section.getValue().entrySet()) {

                results.append(" | ");
                results.append(String.format("%-35s", subSection.getKey()));    // Left align the sub section header

                // If there's some value to print, print that first, and the print anything else
                if (subSection.getValue().length > 0) {
                    results.append(subSection.getValue()[0]);
                    results.append(System.lineSeparator());
                } else {
                    // Otherwise just put nil
                    results.append("nil");
                    results.append(System.lineSeparator());
                }

                // Loop through the remaining values and append the values
                for (int i=1; i<subSection.getValue().length; i++) {
                    results.append(" | ");
                    results.append(String.format("%-35s", ""));
                    results.append(subSection.getValue()[i]);
                    results.append(System.lineSeparator());
                }
            }

            // Finally add the |_ to conclude a section
            results.append(" |_");
            results.append(System.lineSeparator());
            results.append(System.lineSeparator());
        }

        // Now the Independent Variables table
        int columnWidth = 22;
        int columnCount = 7;
        int width       = columnCount*columnWidth - (columnWidth-15);            // Width of table (x columns, y spaces each)
                                                                                 //  - colWidth + 15 because 'Labels' has diff. padding
        // Strings used for formatting (String.format)
        String leftJustify  = "%-15s";
        String rightJustify = "%"  + columnWidth + "s";

        // The dashes (----------) and underscores(_________)
        String fillerDashes       = StringUtils.repeat("-", width+4);
        String fillerUnderscores  = StringUtils.repeat("_", width+4);

        // TODO: Standardized Coefficient
        String[] columnHeaders = new String[] {"Coefficient", "Std. Error", "Deg. of Freedom", "Avg. Residual", "t value", "p value"};

        results.append("Independent Variables/Intercept");
        results.append(System.lineSeparator());
        results.append(" | ");
        results.append(System.lineSeparator());
        results.append(" |");
        results.append(fillerDashes);
        results.append(System.lineSeparator());

        // Append the column headers
        results.append(" | ");
        results.append(String.format(leftJustify, "Labels"));               // Labels is the only column that is left-justified
        for (String header : columnHeaders) results.append(String.format(rightJustify, header));
        results.append("   |");
        results.append(System.lineSeparator());

        results.append(" |");
        results.append(fillerDashes);
        results.append("|");
        results.append(System.lineSeparator());

        // TODO: Calculate Std. Error, Deg. of Freedom, etc. during regression, and INSIDE the CorrelationDataset
        // TODO: Straighten this up: The variable labels are  in correlation dataset, but the coefficients are in the equation?
        // Because the intercept isn't actually 'data' that has a label for the Correlation Dataset, it has to be hardcoded
        results.append(" | ");
        results.append(String.format(leftJustify, "Intercept"));                                        // The word intercept
        results.append(String.format(rightJustify, MathUtils.round(equation.B0, 3)));      // The intercept's actual value
        for (int i=0; i<5; i++) results.append(String.format(rightJustify, "?"));   // Placeholder TODO: REMOVE
        results.append("   |");
        results.append(System.lineSeparator());

        // Now the 'soft-coding'; Loop through each coefficient, append label coefficient, and eventually the other things
        for (int coefficientIndex=0; coefficientIndex < equation.coefficients.numRows(); coefficientIndex++) {
            results.append(" | ");
            results.append(String.format(leftJustify, datasetReference.getIndependentLabel(coefficientIndex)));
            results.append(String.format(rightJustify, MathUtils.round(equation.coefficients.getAt(coefficientIndex, 0), 3)));  // Column vector so col index is always 0
            for (int i=0; i<5; i++) results.append(String.format(rightJustify, "?"));   // Placeholder TODO: REMOVE
            results.append("   |");
            results.append(System.lineSeparator());
        }

        results.append(" |");
        results.append(fillerUnderscores);
        results.append("|");
        results.append(System.lineSeparator());

        System.out.println(results.toString());
    }
}
