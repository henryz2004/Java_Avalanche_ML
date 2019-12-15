import avalanche.neuralnet.nets.NeuralNet;
import avalanche.neuralnet.util.Activation;
import avalanche.neuralnet.util.NeuronLayer;
import avalanche.num.Matrix;

public class DNNTesting {

	public static void main(String[] args) throws Exception{
		// In order to have constant results, we need a constant seed

		NeuralNet testNet = new NeuralNet(Activation.SIGMOID);	// No layers to begin with
		testNet.setLearningRate(1);

		testNet.addLayer(new NeuronLayer(4, 3, 1));
		testNet.addLayer(new NeuronLayer(1, 4, 1));

		testNet.printLayers(true);
		System.out.println();

		// TODO: Biases???
		double[][] trainingIns  = {
				{0,0,1},
				{0,1,1},
				{1,0,1},
				{0,1,0},
				{1,0,0},
				{1,1,1},
				{0,0,0},
		};
		double[] trainingOuts = {0,1,1,1,1,0,0};

		Matrix trainingInputs  = Matrix.from2D(trainingIns);
		Matrix trainingOutputs = Matrix.from1D(trainingOuts).transpose();

		testNet.train(trainingInputs, trainingOutputs, 75000);
		
		System.out.println("Considering a new situation [1,1,0] -> ?: ");
		System.out.println(testNet.think(Matrix.from1D(new double[] {1,1,0})).toCleanString());

		System.out.println();

		System.out.println("Considering a new situation [0,0,1] -> ?: ");
		System.out.println(testNet.think(Matrix.from1D(new double[] {0,0,1})).toCleanString());

		System.out.println();

		System.out.println("Considering a new situation [1,0,1] -> ?: ");
		System.out.println(testNet.think(Matrix.from1D(new double[] {1,0,1})).toCleanString());

		System.out.println();

		System.out.println("Considering a new situation [0, 0, 0] -> ?: ");
		System.out.println(testNet.think(Matrix.from1D(new double[] {0,0,0})).toCleanString());

		System.out.println();

		testNet.printLayers(true);
	}
}
