import avalanche.num.Matrix;

public class MatrixTesting3 {
    public static void main(String[] args) throws Exception {

        Matrix matrix = Matrix.from2D(
                new double[][] {
                        {1,3},
                        {5,7}
                }
        );


        long start = System.currentTimeMillis();
        System.out.println("Inverse: " + matrix.calculateInverse().toCleanString());
        long end   = System.currentTimeMillis();

        System.out.println(end - start);
    }
}