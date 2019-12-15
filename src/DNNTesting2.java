import avalanche.neuralnet.nets.NeuralNet;
import avalanche.neuralnet.util.Activation;
import avalanche.neuralnet.util.NeuronLayer;
import avalanche.num.Matrix;

public class DNNTesting2 {

    public static void main(String[] args) throws Exception{
        // In order to have constant results, we need a constant seed

        NeuralNet testNet = new NeuralNet(Activation.SIGMOID);	// No layers to begin with
        testNet.setLearningRate(1);

        testNet.addLayer(new NeuronLayer(4, 7, 1));
        testNet.addLayer(new NeuronLayer(1, 4, 1));

        testNet.printLayers(true);
        System.out.println();

        double[][] trainingIns  = {
                {0,0,0,0,0,0,0},
                {1,1,1,1,1,1,1},
                {0,1,1,1,1,0,0},
                {0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4},
                {0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6},
                {0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25},
                {0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8}
        };
        double[] trainingOuts = {1,1,0,0,0,1,1};

        Matrix trainingInputs  = Matrix.from2D(trainingIns);
        Matrix trainingOutputs = Matrix.from1D(trainingOuts).transpose();

        testNet.train(trainingInputs, trainingOutputs, 60000);

        // Try to serialize the neural net
        System.out.println("Dumping net");
        NeuralNet.saveNet(testNet, "NeuralNet.ava");

        System.out.println("Reading net");
        NeuralNet readNet = NeuralNet.loadNet("NeuralNet.ava");

        System.out.println();

        if (readNet == null) throw new NullPointerException();

        System.out.println("Considering [0.1, 0.05, 0.009, 0.2, 0.06, 0.12, 0.11] -> ?");
        System.out.println(readNet.think(Matrix.from1D(new double[]{0.1, 0.05, 0.009, 0.2, 0.06, 0.12, 0.11})));

        System.out.println("Considering [0.9, 0.978, 0.999978, 0.86, 0.91, 0.95, 0.98] -> ?");
        System.out.println(readNet.think(Matrix.from1D(new double[]{0.9, 0.978, 0.999978, 0.86, 0.91, 0.95, 0.98})));

        System.out.println("Considering [0.1, 0.978, 0.999978, 0.86, 0.91, 0.12, 0.11] -> ?");
        System.out.println(readNet.think(Matrix.from1D(new double[]{0.1, 0.978, 0.999978, 0.86, 0.91, 0.12, 0.11})));

        System.out.println("Considering [0.1, 0.978, 0.6, 0.86, 0.91, 0.2, 0.45] -> ?");
        System.out.println(readNet.think(Matrix.from1D(new double[] {0.1, 0.978, 0.6, 0.86, 0.91, 0.2, 0.45})));

        System.out.println("Considering [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5] -> ?");
        System.out.println(readNet.think(Matrix.from1D(new double[] {0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5})));

        System.out.println("Considering [0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6] -> ?");
        System.out.println(readNet.think(Matrix.from1D(new double[] {0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6})));

    }
}
