package avalanche.qlearning.storage;

import avalanche.util.annotations.Dangerous;

import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

public class MassStore {
    // Efficient storage for the masses.
    // https://docs.google.com/document/d/1tAmydt6xXoT12trtgN0pPluvyewWTwgQvB6L8Eywt6Y/edit#

    private int rows;
    private int cols;

    private double defaultValue;
    private double maximumValue;        // Used for quick normalization

    private Map<ArrayCoord, Double> uniqueValues;

    // Public constructors
    public MassStore() {
        this((double) 0);
    }
    public MassStore(double defVal) {
        rows = 0;
        cols = 0;
        defaultValue = defVal;
        maximumValue = defaultValue;
        uniqueValues = new HashMap<>();
    }

    public void addRow() {
        rows++;
    }
    public void addCol() {
        cols++;
    }

    public int numRows() {
        return rows;
    }
    public int numCols() {
        return cols;
    }

    // Getters and Setters
    public void clear() {
        int rows = 0;
        int cols = 0;
        uniqueValues.clear();
    }
    public void set(int row, int col, double value) {
        set(new int[] {row, col}, value);
    }
    public void set(int[] coord, double value) {
        // Check out of bounds
        if (coord[0] >= rows || coord[1] >= cols) {
            throw new ArrayIndexOutOfBoundsException("Coordinate out of bounds");
        }

        if (value != defaultValue) uniqueValues.put(new ArrayCoord(coord), value);
        if (value > maximumValue) maximumValue = value;     // Update maximum value
    }
    public void setMassValue(double defVal) {
        defaultValue = defVal;

        // Update the maximum value (maybe)
        if (defVal > maximumValue) {
            maximumValue = defVal;
        }
    }
    public void addRowList(List<Double> rowList) {

        // First add the row
        addRow();
        int rowIndex = rows-1;

        // Then add the contents
        for (int col=0; col < cols; col++) {
            set(rowIndex, col, rowList.get(col));
        }

    }

    @Dangerous
    public void addToRow(int row, double value) {
        addCol();
        set(row, cols-1, value);
    }

    public double get(int row, int col) {
        return get(new int[] {row, col});
    }
    public double get(int[] coord) {
        // Check out of bounds
        if (coord[0] >= rows || coord[1] >= cols) {
            throw new ArrayIndexOutOfBoundsException("Coordinate out of bounds");
        }

        // If this coordinate is unique then return the unique value. Otherwise, just return the default value
        ArrayCoord coordinate = new ArrayCoord(coord);
        return uniqueValues.getOrDefault(coordinate, defaultValue);
    }

    // Normalizer
    public void normalize() {
        // Use an iterator and iterate through all the non-default values
        Iterator<Map.Entry<ArrayCoord, Double>> entryIterator = uniqueValues.entrySet().iterator();

        while (entryIterator.hasNext()) {
            Map.Entry<ArrayCoord, Double> entry = entryIterator.next();
            uniqueValues.put(entry.getKey(), entry.getValue()/maximumValue);
        }
    }
}
