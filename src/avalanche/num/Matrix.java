package avalanche.num;

import avalanche.num.util.MathUtils;
import avalanche.util.annotations.Dangerous;

import java.io.Serializable;
import java.util.Arrays;
import java.util.InputMismatchException;
import java.util.List;
import java.util.Random;
import java.util.concurrent.ThreadLocalRandom;

/**
 * @version 1.2
 * copy() temporarily deprecated because it does not work.
 *
 * 'Lightweight' implementation of matrices
 */
public class Matrix implements Serializable {

	private int rows;
	private int cols;
	
	private double[][] values;	// 2D array of the values in this Matrix

	private static Random rng;	// Used for seed-setting

	// Static initializer block
	static {
		rng = new Random();
	}

	// Private constructor
	private Matrix(int r, int c, double[][] rowValues) {
		
		rows = r;
		cols = c;
		
		values = rowValues;
	}
	
	// Factory methods
	public static Matrix fillEmpty(int rows, int cols, double defaultValue) {
		// Initialize a new Matrix with all values set to the parameterized default value
		double[][] matrixValues = new double [rows][cols];
		
		// Fill matrixValues with defaultValue
		for (double[] matrixRow : matrixValues) {
			for (int i=0; i<matrixRow.length; i++) {
				matrixRow[i] = defaultValue;
			}
		}
		
		return new Matrix(rows, cols, matrixValues);
	}
	public static Matrix fillRandom(int rows, int cols) {
		return fillRandom(rows, cols, 0, 1);
	}
	public static Matrix fillRandom(int rows, int cols, double start, double end) {
		// Fills a matrix with random values between start and end

		double[][] matrixValues = new double [rows][cols];
		
		for (double[] matrixRow : matrixValues) {
			for (int i=0; i<matrixRow.length; i++) {
				matrixRow[i] = ThreadLocalRandom.current().nextDouble(start, end);
			}
		}
		
		return new Matrix(rows, cols, matrixValues);
	}
	public static Matrix fillRandom(int rows, int cols, double[] range1, double[] range2) {
		// Fill Matrix with values from either range 1 or range 2

		double[][] matrixValues = new double[rows][cols];

		for (double[] matrixRow : matrixValues) {
			for (int i=0; i<matrixRow.length; i++) {
				if (ThreadLocalRandom.current().nextBoolean()) {
					matrixRow[i] = ThreadLocalRandom.current().nextDouble(range1[0], range1[1]);
				} else {
					matrixRow[i] = ThreadLocalRandom.current().nextDouble(range2[0], range2[1]);
				}
			}
		}

		return new Matrix(rows, cols, matrixValues);
	}
	public static Matrix fillRandom(int rows, int cols, double[] range1, double[] range2, long seed) {
		// Same as the normal version, except uses rng
		rng.setSeed(seed);

		double[][] matrixValues = new double[rows][cols];

		for (double[] matrixRow : matrixValues) {
			for (int i=0; i<matrixRow.length; i++) {
				if (rng.nextBoolean()) {
					matrixRow[i] = rngDoubleInRange(range1[0], range1[1]);
				} else {
					matrixRow[i] = rngDoubleInRange(range2[0], range2[1]);
				}
			}
		}

		//rng.setSeed(System.currentTimeMillis());		TODO: Add option to reset the seed

		return new Matrix(rows, cols, matrixValues);
	}
	public static Matrix from2D(double[][] inArray) {
		// Convert a 2D array to a Matrix
		return new Matrix(inArray.length, inArray[0].length, inArray);
	}
	public static Matrix from2D(List<double[]> inList) {
		return new Matrix(inList.size(), inList.get(0).length, (double[][]) inList.toArray());
	}
	public static Matrix from1D (double[] inArray) {
		return Matrix.from2D(new double[][] {inArray});
	}
	public static Matrix fromRVectors(List<Matrix> rowVectors) {
		int r = rowVectors.size();
		int c = rowVectors.get(0).cols;

		double[][] array = new double[r][c];
		for (int i=0; i<array.length; i++) {
			array[i] = rowVectors.get(i).toArray()[0];
		}

		return new Matrix(r, c, array);
	}
	public static Matrix identityMatrix(int size) {
		// Returns an identity matrix of dimensions [size, size]
		Matrix idMatrix = Matrix.fillEmpty(size, size, 0);
		
		for (int i=0; i<idMatrix.numRows(); i++) {
			idMatrix.setAt(i, i, 1);
		}
		return idMatrix;
	}
	public Matrix transpose() {
		
		// Make a new matrix with flipped rows and cols
		Matrix transposed = Matrix.fillEmpty(cols, rows, 0);
		
		// Rows of this matrix is equal to the columns of that
		for (int rowIndex=0; rowIndex<values.length; rowIndex++) {
			double[] row = values[rowIndex];

			for (int col=0; col<row.length; col++) {
				transposed.setAt(col, rowIndex, row[col]);
			}
		}
		
		return transposed;
	}

	/**
	 * @deprecated because bugs.
	 */
	@Dangerous
	@Deprecated
	// FIXME
	public Matrix copy() {
		return Matrix.from2D(values);
	}
	public void overwrite(Matrix matrix) {
		// Overwrites the current matrix with the input matrix
		// Not a factory method per se, but it does change the matrix entirely
		rows = matrix.numRows();
		cols = matrix.numCols();
		values = matrix.toArray();
	}
	
	// Other methods
	// Getters.
	public int numRows() {
		return rows;
	}
	public int numCols() {
		return cols;
	}
	public int[] shape() {
		return new int[] {rows, cols};
	}
	public double getAt(int row, int col) {
		// Gets the value at [row][col]
		return values[row][col];
	}
	public double[] flatten() {
		double[] flatArray = Arrays.stream(values)
				.flatMapToDouble(Arrays::stream)
				.toArray();
		return flatArray;
	}
	public double[][] toArray() {
		return values;
	}
	public Matrix slice(int[] rowSlice, int[] colSlice) {
		// TODO: Add increment, negative slicing
		// [Inclusive, exclusive)
		
		// Array for sliced rows
		double[][] slicedRows = Arrays.copyOfRange(values, rowSlice[0], rowSlice[1]);//new double[rowSlice[1]-rowSlice[0]][cols];
		double[][] slicedCols = new double[slicedRows.length][colSlice[1]-colSlice[0]];
		
		for (int r=0; r < slicedRows.length; r++) {
			slicedCols[r] = Arrays.copyOfRange(slicedRows[r], colSlice[0], colSlice[1]);
		}
		
		return Matrix.from2D(slicedCols);
	}
	public Matrix getRow(int rowIndex) {

		double[][] listResult = new double[1][values[0].length];

		for (int c=0; c < values[0].length; c++) {
			listResult[0][c] = values[rowIndex][c];
		}

		return Matrix.from2D(listResult);
	}
	public Matrix getRows(int from, int to) {

		double[][] listResult = new double[to-from][values[0].length];

		for (int c=0; c < values[0].length; c++) {
			for (int r=from; r < to; r++) {
				listResult[r-from][c] = values[r][c];
			}
		}

		return Matrix.from2D(listResult);
	}
	public Matrix getColumn(int colIndex) {

		double[][] listResult = new double[values.length][1];

		for (int r=0; r < values.length; r++) {
			listResult[r][0] = values[r][colIndex];
		}

		return Matrix.from2D(listResult);
	}
	public Matrix getColumns(int from, int to) {
		// [Inclusive, exclusive)

		double[][] listResult = new double[values.length][to-from];

		for (int r=0; r < values.length; r++) {
			for (int c=from; c < to; c++) {
				listResult[r][c-from] = values[r][c];
			}
		}

		return Matrix.from2D(listResult);
	}
	public Matrix exceptColumn(int columnIndex)  {
		// Same matrix EXCEPT for that column

		// Last row exempted
		if (columnIndex == -1) {
			columnIndex = values[0].length-1;		// FIXME: Length 0 values
		}

		double[][] listResult = new double[values.length][values[0].length-1];

		for (int r=0; r < values.length; r++) {

			int columnInsertionIndex = 0;

			for (int c=0; c < values[0].length-1; c++) {
				if (c != columnIndex) {
					listResult[r][columnInsertionIndex] = values[r][c];
					columnInsertionIndex++;
				}
			}
		}

		return Matrix.from2D(listResult);
	}

	// Setters
	public void setAt(int row, int col, double newValue) {
		values[row][col] = newValue;
	}
	public void fillWith(double newValue)  {
		for (double[] row : values) {
			for (int i=0; i<row.length; i++) {
				row[i] = newValue;
			}
		}
	}
	
	// Action-ers
	// Static methods all use the non-static actioners
	public static double determinantOf(Matrix matrix) {
		return matrix.calculateDeterminant();
	}
	public static Matrix inverseOf(Matrix matrix) {
		return matrix.calculateInverse();
	}
	public static Matrix dotProduct(Matrix matrix, Matrix otherMatrix) {
		return matrix.dotProduct(otherMatrix);
	}
	public static Matrix multiply(Matrix matrix, Matrix otherMatrix) {
		return matrix.multiply(otherMatrix);
	}
	public static Matrix add(Matrix matrix, Matrix otherMatrix) throws Exception{
		return matrix.add(otherMatrix);
	}
	public static Matrix sub(Matrix matrix, Matrix otherMatrix) throws Exception{
		return matrix.sub(otherMatrix);
	}
	public static Matrix scalarProduct(double scalar, Matrix matrix) {
		return matrix.scalarProduct(scalar);
	}
	public static Matrix scalarSum(double scalar, Matrix matrix) {
		return matrix.scalarSum(scalar);
	}
	public static Matrix scalarSub(double scalar, Matrix matrix) {
		return matrix.scalarSub(scalar);
	}
	public static Matrix subFromScalar(Matrix matrix, double scalar) { return matrix.subFromScalar(scalar); }

	public double calculateDeterminant() {
		// MAKE SURE THAT THIS MATRIX IS A SQUARE
		if (rows != cols) {
			throw new ArithmeticException(
					"Only square matrices have determinants. This matrix's dimensions are: " + rows + " by " + cols
			);
		}

		// If the current matrix is 2x2, then we can
		// easily find the determinant using ad - bc
		// cols is presumed to be equal to rows
		if (rows == 2) {
			return values[0][0]*values[1][1] - values[0][1]*values[1][0];
		}

		return recursiveDeterminant(values);
	}
	public Matrix calculateInverse() {

		// A 2x2 matrix is a special case
		if (rows == 2) {
			Matrix intermediateMatrix = Matrix.from2D(
					new double[][] {
							{values[1][1], -values[0][1]},
							{-values[1][0], values[0][0]}
					}
			);
			return intermediateMatrix.scalarProduct(1/recursiveDeterminant(values));
		}

		// Calculate Matrix of Minors, and determinant while at at
		double[][] matrixOfMinors = new double[rows][cols];
		double     determinant = 0;
		int        multiplier  = 1;

		for (int r=0; r < rows; r++) {
			for (int c=0; c < cols; c++) {
				double localDeterminant = recursiveDeterminant(ignoreRowAndColumn(values, r, c));
				matrixOfMinors[r][c] = localDeterminant;
				if (r==0) {
					determinant += multiplier * values[r][c] * localDeterminant;
					multiplier *= -1;
				}
			}
		}

		// If the determinant is 0, throw an exception to tell the user that the matrix is not invertible
		if (determinant == 0) {
			throw new ArithmeticException("Input matrix is not invertible.");
		}

		// Calculate the matrix of cofactors
		double[][] matrixOfCofactors = calculateCofactors(matrixOfMinors);

		// Finally multiply the adjugate by 1/determinant to get the inverse
		return Matrix.from2D(matrixOfCofactors).transpose().scalarProduct(1/determinant);
	}
	public Matrix dotProduct(Matrix otherMatrix) {
		
		// First check if the 2 Matrices are compatible
		if (cols != otherMatrix.numRows()) {
			throw new InputMismatchException("Number of columns ("+cols+") does not match the number of rows ("+otherMatrix.numRows()+")");
		}
		
		double[][] listResult = new double[rows][otherMatrix.numCols()];
		
		// For every row in our matrix, multiply it to every column in that matrix
		for (int matrixRIndex=0; matrixRIndex < rows; matrixRIndex++) {
			double[] matrixRow = values[matrixRIndex];
			
			for (int matrixCIndex=0; matrixCIndex < otherMatrix.numCols(); matrixCIndex++) {
				
				// Multiply and sum
				double sum = 0;
				for (int valueIndex=0; valueIndex<matrixRow.length; valueIndex++) {
					sum += matrixRow[valueIndex] * otherMatrix.getAt(valueIndex, matrixCIndex);
				}
				
				listResult[matrixRIndex][matrixCIndex] = sum;	// Finally add it back to our 2D Array Matrix
			}
		}
		
		return Matrix.from2D(listResult);
	}
	public Matrix multiply(Matrix otherMatrix) {
		// Simple multiply
		
		// Check compatibility
		if (cols != otherMatrix.numCols() ||
				rows != otherMatrix.numRows()) {
			throw new InputMismatchException(
				"Number of columns and rows must be equivalent for simple multiplication." + System.lineSeparator() +
				rows + " rows and " + otherMatrix.numRows() + " rows may not match." + System.lineSeparator() +
				cols + " cols and " + otherMatrix.numCols() + " cols may not match."
			);
		}
		
		double[][] listResult = new double[rows][cols];
		
		for (int r=0; r < listResult.length; r++) {
			for (int c=0; c < listResult[0].length; c++) {
				listResult[r][c] = values[r][c] * otherMatrix.getAt(r, c);				
			}
		}
		
		return Matrix.from2D(listResult);
	}
	public Matrix add(Matrix otherMatrix) throws InputMismatchException{
		
		if (rows != otherMatrix.numRows() || cols != otherMatrix.numCols()) {
			throw new InputMismatchException(
				"The two matricies must be of same dimensions." + System.lineSeparator() +
				rows + " rows and " + otherMatrix.numRows() + " rows may not match." + System.lineSeparator() +
				cols + " cols and " + otherMatrix.numCols() + " cols may not match."
			);
		}
		
		double[][] listResult = new double[rows][cols];
		
		// Sum each corresponding element
		for (int rowIndex=0; rowIndex<values.length; rowIndex++) {
			for (int colIndex=0; colIndex<values[0].length;  colIndex++) {
				listResult[rowIndex][colIndex] = values[rowIndex][colIndex] + otherMatrix.getAt(rowIndex, colIndex);
			}
		}
		
		return Matrix.from2D(listResult);
	}
	public Matrix sub(Matrix otherMatrix) throws InputMismatchException{
		
		if (rows != otherMatrix.numRows() || cols != otherMatrix.numCols()) {
			throw new InputMismatchException("The two matrices must be of same dimensions; ("+ rows + ", " + cols + ") and ("+ otherMatrix.numRows() + ", " + otherMatrix.numCols() + ")");
		}
		
		double[][] listResult = new double[rows][cols];
		
		// Sum each corresponding element
		for (int rowIndex=0; rowIndex<values.length; rowIndex++) {
			for (int colIndex=0; colIndex<values[0].length;  colIndex++) {
				listResult[rowIndex][colIndex] = values[rowIndex][colIndex] - otherMatrix.getAt(rowIndex, colIndex);
			}
		}
		
		return Matrix.from2D(listResult);
	}
	public Matrix scalarProduct(double scalar) {
		
		double[][] listResult = new double[rows][cols];
		
		for (int i=0; i<values.length; i++) {
			for (int j=0; j<values[0].length; j++) {
				listResult[i][j] = values[i][j] * scalar;
			}
		}
		
		return Matrix.from2D(listResult);
	}
	public Matrix scalarSum(double scalar) {

		double[][] listResult = new double[rows][cols];
		
		for (int i=0; i<values.length; i++) {
			for (int j=0; j<values[0].length; j++) {
				listResult[i][j] = values[i][j] + scalar;
			}
		}
		
		return Matrix.from2D(listResult);
	}
	public Matrix scalarSub(double scalar) {

		double[][] listResult = new double[rows][cols];
		
		for (int i=0; i<values.length; i++) {
			for (int j=0; j<values[0].length; j++) {
				listResult[i][j] = values[i][j] - scalar;
			}
		}
		
		return Matrix.from2D(listResult);
	}
	public Matrix subFromScalar(double scalar) {

		double[][] listResult = new double[rows][cols];

		for (int i=0; i<values.length; i++) {
			for (int j=0; j<values[0].length; j++) {
				listResult[i][j] = scalar - values[i][j];
			}
		}

		return Matrix.from2D(listResult);
	}

	public double sumUp() {
		double sum = 0;
		for (double[] row : values) {
			for (double element : row) {
				sum += element;
			}
		}
		return sum;
	}
	public double sumUpAbs() {
		double sum = 0;
		for (double[] row : values) {
			for (double element : row) {
				sum += Math.abs(element);
			}
		}
		return sum;
	}
	public double findMax() {
		double max = Double.NEGATIVE_INFINITY;
		for (double[] row : values) {
			for (double element : row) {
				if (element > max) max = element;
			}
		}
		return max;
	}
	public double findMean() {
		double sum = sumUp();
		return sum/(rows*cols);
	}

	public Matrix useExpression(Expression expression) {
		// Passes each value through the expression.evaluate and puts the output into the return Matrix
		
		double[][] listResult = new double[rows][cols];
		
		for (int r=0; r < rows; r++) {
			for (int c=0; c < cols; c++) {
				listResult[r][c] = expression.evaluate(values[r][c]);
			}
		}
		
		return Matrix.from2D(listResult);
	}

	// Helper functions
	private static double rngDoubleInRange(double min, double max) {
		return min + (max - min) * rng.nextDouble();
	}
	private static double recursiveDeterminant(double[][] matrix) {
		// Base case: 2x2 matrix. Input matrix should always be a square, therefore we can assume that
		// # rows = # cols
		if (matrix.length == 2) {
			return matrix[0][0]*matrix[1][1] - matrix[0][1]*matrix[1][0];
		}

		int multiplier = 1;	// Tells us whether to subtract or add
		double determinant = 0;
		for (int topNumberCol=0; topNumberCol < matrix.length; topNumberCol++) {
			double topNumber = matrix[0][topNumberCol];
			determinant +=
					multiplier					// Determines whether we're adding or subtracting
							* topNumber			// a or b or c etc. times the determinant of the rest of the matrix
							* recursiveDeterminant(ignoreRowAndColumn(matrix, 0, topNumberCol));
			multiplier *= -1;	// Flip the multiplier
		}
		return determinant;
	}
	private static double[][] calculateCofactors(double[][] matrixOfMinors) {
		// Apply a checkerboard pattern
		double[][] matrixOfCofactors = new double[matrixOfMinors.length][matrixOfMinors[0].length];
		int multiplier = 1;
		for (int r=0; r < matrixOfMinors.length; r++) {
			for (int c=0; c < matrixOfMinors[0].length; c++) {
				matrixOfCofactors[r][c] = multiplier * matrixOfMinors[r][c];
				multiplier *= -1;
			}
		}
		return matrixOfCofactors;
	}
	private static double[][] ignoreRowAndColumn(double[][] matrix, int ignoreRow, int ignoreCol) {
		double[][] filtered = new double[matrix.length-1][matrix[0].length-1];
		int i = 0;
		int j = 0;
		for (int r=0; r < matrix.length; r++) {
			if (r == ignoreRow) continue;
			for (int c=0; c < matrix[0].length; c++) {
				if (c == ignoreCol) continue;
				filtered[i][j] = matrix[r][c];
				j++;
			}
			i++;
			j=0;
		}
		return filtered;
	}

	@Override
	public boolean equals(Object other) {
		// GOOD comparison
		if (!(other instanceof Matrix)) return false;
		return (Arrays.deepEquals(values, ((Matrix) other).toArray()));
	}

	// toString and toCleanString
	@Override
	public String toString() {
		return Arrays.deepToString(values);
	}
	public String toCleanString() {
		double[][] roundedVals = new double[rows][cols];
		for (int r=0; r<rows; r++) {
			for (int c=0; c<cols; c++) {
				roundedVals[r][c] = MathUtils.round(values[r][c], 2);
			}
		}
		return Arrays.deepToString(roundedVals);
	}
}
