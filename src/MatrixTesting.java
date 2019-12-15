import avalanche.num.Expression;
import avalanche.num.Matrix;

public class MatrixTesting {
	public static void main(String[] args) throws Exception {
		
		double[][] matrixBlueprint = {
				{1,2,3},
				{4,5,6},
				{7,8,9}
		};
		
		Matrix fromBlueprint = Matrix.from2D(matrixBlueprint);
		Matrix logiBlueprint = fromBlueprint.useExpression(new sig());
		Matrix filledWith2s  = Matrix.fillEmpty(3, 3, 2);

		// Block commented out tests are successful tests
		// Line commented out tests are questionable tests

		/*System.out.println(Matrix.multiply(fromBlueprint, filledWith1s));
		System.out.println(fromBlueprint.multiply(filledWith1s));
		System.out.println(Matrix.sub(fromBlueprint, filledWith1s));
		System.out.println(fromBlueprint.sub(filledWith1s));
		System.out.println(Matrix.scalarSum(1, fromBlueprint));
		System.out.println(Matrix.scalarSum(-1, fromBlueprint));
		System.out.println(fromBlueprint.useExpression(new sig()));
		System.out.println(fromBlueprint.useExpression(new sig()).useExpression(new sigd()));
		System.out.println(logiBlueprint.multiply(logiBlueprint.subFromScalar(1)));*/
		/*System.out.println(logiBlueprint);*/
		System.out.println(fromBlueprint);
		System.out.println(fromBlueprint.transpose());
	}
	static class sig implements Expression {
		public double evaluate(double input) {
			return 1/(1+Math.pow(2.7183, -input));
		}
	}
	static class sigd implements Expression {
		public double evaluate(double input) {
			return input * (1 - input);
		}
	}
}