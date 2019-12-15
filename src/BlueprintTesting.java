import avalanche.neuralnet.nets.NeuralNet;
import avalanche.neuralnet.util.Activation;
import avalanche.neuralnet.util.architecture.NeuralBlueprint;
import avalanche.num.Matrix;

@Deprecated
public class BlueprintTesting {

    public static void main(String args[]) {

        NeuralBlueprint blueprint = new NeuralBlueprint(3, 1);
        NeuralNet       neuralNet = blueprint.constructRandomNetwork();

        NeuralNet       manualNet = new NeuralNet(Activation.SIGMOID);
        manualNet.addLayer(4, 3);
        manualNet.addLayer(1, 4);

        double[][] trainingIns  = {
                {0,0,1},	// 0
                {0,1,1},	// 1
                {1,0,1},	// 1
                {0,1,0},	// 1
                {1,0,0},	// 1
                {1,1,1},	// 0
                {0,0,0},	// 0
        };
        double[] trainingOuts = {0,1,1,1,1,0,0};

        Matrix trainingInputs  = Matrix.from2D(trainingIns);
        Matrix trainingOutputs = Matrix.from1D(trainingOuts).transpose();

        try {
            neuralNet.printLayers(true);
            System.out.println();
            manualNet.printLayers(true);
            System.out.println();
            System.out.println(neuralNet.think(trainingInputs).toCleanString());
            System.out.println(manualNet.think(trainingInputs).toCleanString());
            System.out.println();

            neuralNet.train(trainingInputs, trainingOutputs, 150000);

            System.out.println("Considering a new situation [1,1,0] -> ?: ");
            System.out.println(neuralNet.think(Matrix.from1D(new double[] {1,1,0})/*, new int[] {0, 1}*/).toString());
            System.out.println(neuralNet.think(Matrix.from1D(new double[] {1,0,1})/*, new int[] {0, 1}*/).toString());

            neuralNet.printLayers(true);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
