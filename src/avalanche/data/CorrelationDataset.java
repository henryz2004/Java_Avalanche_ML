package avalanche.data;

import avalanche.util.ArrayUtils;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class CorrelationDataset {
    // Class that is similar to Dataset, except there can be
    // Multiple dependent and independent variables
    // Used for Multiple Linear Regression
    // TODO: Degrees of freedom?

    public List<double[]> independentData;
    public List<double[]> dependentData;

    public List<String> independentLabels;
    public List<String> dependentLabels;

    public Dataset internalDataset;    // This will refer to the real data stored in this dataset

    // Constructor will be able to accept Dataset as input, and the LAST
    // Value (coordinate) will be considered as the dependent variable
    public CorrelationDataset() {
        independentData = new ArrayList<>();
        dependentData   = new ArrayList<>();

        independentLabels = new ArrayList<>();
        dependentLabels   = new ArrayList<>();

        internalDataset = new Dataset();
    }
    public CorrelationDataset(Dataset dataset) {
        // Independent data is everything up to the last column; dependent is last column
        independentData = ArrayUtils.getColumnsAsList(dataset.data, 0, dataset.data.get(0).length-1);
        dependentData   = ArrayUtils.getColumnsAsList(dataset.data, dataset.data.get(0).length-1, dataset.data.get(0).length);

        independentLabels = new ArrayList<>();
        dependentLabels   = new ArrayList<>();

        internalDataset = dataset;
    }

    // Labels are basically the 'name' of the variable
    public void setLabels(List<String> xLabels, List<String> yLabels) {
        independentLabels = xLabels;
        dependentLabels   = yLabels;
    }
    public void labelIndependents(List<String> yLabels) {
        independentLabels = yLabels;
    }
    public void labelDependents(List<String> xLabels) {
        dependentLabels   = xLabels;
    }

    // When adding data, we must add it to not only our Lists,
    // but also to our internal dataset
    public void addData(double[] newData) {
        // As usual, the 'dependent' variable is assumed to be the last variable
        independentData.add(Arrays.copyOfRange(newData, 0, newData.length-1));
        dependentData.add(Arrays.copyOfRange(newData, newData.length-1, newData.length));

        internalDataset.addData(newData);
    }
    public void addData(double[] independent, double[] dependent) {
        independentData.add(independent);
        dependentData.add(dependent);

        internalDataset.addData(ArrayUtils.concatenate(independent, dependent));
    }
    public void addBulkData(List<double[]> newBulkData) {
        for (double[] newData : newBulkData) {
            addData(newData);
        }
    }
    public void addBulkData(double[][] newBulkData) {
        // Exactly the same as addBulkData(List<double[]>);
        for (double[] newData : newBulkData) {
            addData(newData);
        }
    }
    public void addBulkData(List<double[]> independents, List<double[]> dependents) {
        for (int i=0; i<independents.size(); i++) {
            addData(independents.get(i), dependents.get(i));
        }
    }
    public void addBulkData(double[][] independents, double[][] dependents) {
        // Exactly the same as addBulkData(List<double[]>, List<double[])
        for (int i=0; i<independents.length; i++) {
            addData(independents[i], dependents[i]);
        }
    }

    public String getIndependentLabel(int index) {
        if (independentLabels.size() <= index) {
            return "Unnamed";   // No label
        }
        return independentLabels.get(index);
    }
    public String getDependentLabel(int index) {
        if (dependentLabels.size() <= index) {
            return "Unnamed";   // No label
        }
        return dependentLabels.get(index);
    }
}
