import avalanche.num.Matrix;

public class MatrixTesting2 {
	public static void main(String[] args) throws Exception {
		
		Matrix matrix = Matrix.identityMatrix(5);

		// BE VERY CAREFUL; THE TIME IT TAKES TO CALCULATE THE DETERMINANT INCREASES INSANELY FAST
		long start = System.currentTimeMillis();
		System.out.println("Determinant: " + matrix.calculateDeterminant());
		long end   = System.currentTimeMillis();

		System.out.println(end - start);
	}
}