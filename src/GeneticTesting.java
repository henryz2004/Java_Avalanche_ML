import avalanche.neuralnet.nets.GeneticAlgorithm;
import avalanche.neuralnet.nets.NeuralNet;
import avalanche.neuralnet.util.architecture.LayerArchitecture;
import avalanche.neuralnet.util.architecture.NetArchitecture;
import avalanche.neuralnet.util.fitness.FitnessFunction;
import avalanche.num.Matrix;

import java.io.File;
import java.io.FileInputStream;
import java.io.ObjectInputStream;
import java.util.Arrays;
import java.util.List;

public class GeneticTesting {

    private static double[][] trainingIns  = {
            {0,0,1},	// 0
            {0,1,1},	// 1
            {1,0,1},	// 1
            {0,1,0},	// 1
            {1,0,0},	// 1
            {1,1,1},	// 0
            {0,0,0},	// 0
    };
    static double[] trainingOuts = {0,1,1,1,1,0,0};

    static Matrix trainingInputs  = Matrix.from2D(trainingIns);
    static Matrix trainingOutputs = Matrix.from1D(trainingOuts).transpose();

    static NeuralNet loaded;

    static {
        try {
            FileInputStream fOi = new FileInputStream(new File("NeuralNet.ava"));
            ObjectInputStream oi = new ObjectInputStream(fOi);

            loaded = (NeuralNet) oi.readObject();

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public static void main(String[] args) throws Exception{

        // Make a architecture for our net because inputs and outputs must be compatible
        NetArchitecture blueprint = new NetArchitecture();
        blueprint.addLayer(new LayerArchitecture(4, 3));
        blueprint.addLayer(new LayerArchitecture(1, 4));

        FitnessFunction ff = new NetFitness();

        // Train the net to the same problem DNNTesting was faced with
        List<NeuralNet> nets = GeneticAlgorithm.train(
                blueprint,
                ff,
                0.1,
                0.5,
                15,
                50
        );

        // Find the best net
        NeuralNet net      = nets.get(0);
        double bestFitness = ff.fitness(net);

        for (int i=1; i<nets.size(); i++) {
            NeuralNet thisNet  = nets.get(i);
            double thisFitness = ff.fitness(thisNet);
            if (thisFitness > bestFitness) {
                net = thisNet;
                bestFitness = thisFitness;
            }
        }

        net.printLayers(true);

        System.out.println(net.think(trainingInputs).toString());
    }

    static class NetFitness implements FitnessFunction {
        @Override
        public double fitness(NeuralNet net) {
            // TODO: Use neural net to score neural net XD

            try {
                Matrix netOutput = net.think(trainingInputs);
                double[] error   = trainingOutputs.sub(netOutput).flatten();

                double totalError = 0;
                for (double errorBit : error) {
                    totalError += Math.abs(errorBit);
                }

                // The closer the error is to 0, the fitter the net
                // Times 10 in order to amplify the fitness
                double totalGoodness = (1/totalError) * 10;

                // However, we must avoid local maxima by penalizing results that resemble all 0's or all 1's
                // use loaded to see if it's hitting a local maxima
                Matrix loadedThought = loaded.think(netOutput.transpose());

                if (loadedThought.getAt(0, 0) > 0.5) {
                    System.out.print("[PENALIZED] " + loadedThought.getAt(0,0) + "\t");
                    totalGoodness /= 2;     // Penalization
                }

                System.out.println(
                        "NetOut " + netOutput.toCleanString()
                                + "\tExpected out: " + Arrays.toString(trainingOuts)
                                + "  Fitness: " + totalGoodness);

                return totalGoodness;

            } catch (Exception e) {
                e.printStackTrace();
                return 0;
            }
        }
    }
}
