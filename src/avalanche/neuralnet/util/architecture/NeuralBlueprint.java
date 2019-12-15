package avalanche.neuralnet.util.architecture;

import avalanche.neuralnet.nets.NeuralNet;
import avalanche.neuralnet.util.Activation;
import avalanche.neuralnet.util.NeuronLayer;

import java.util.concurrent.ThreadLocalRandom;

/**
 * This is used inside GeneticAlgorithm to help encapsulate a simple
 * architecture for a neural net, consisting of number of inputs and
 * number of outputs
 *
 * TODO: FIX- The weights are initialized strangely
 *
 * @deprecated temporarily
 */
@Deprecated
public class NeuralBlueprint {

    public int numInputs;
    public int numOutputs;

    public NeuralBlueprint(int ins, int outs) {
        numInputs  = ins;
        numOutputs = outs;
    }

    public NeuralNet constructRandomNetwork() {
        // Construct a NeuralNet with random hidden layers and # of hidden neurons
        // and random learning rate

        NeuralNet net = new NeuralNet(Activation.SIGMOID);  // For now just stick with Sigmoid. TANH also works, but sigmoid is safest for now

        // TODO: USER DEFINED HIDDEN LAYER COUNT AND NEURONS PER LAYER COUNT
        // Neural net with up to 2 hidden layers and up to numInputs * 2 + 1 neurons per layer
        // and learning rate from 0.01 to 1.1
        int maxNeuronsPerLayer  = numInputs * 2 + 1;
        int numHiddenLayers     = ThreadLocalRandom.current().nextInt(3);
        double[] learningRateRange = {0.01, 1.1};

        // Set the random learning rate
        net.setLearningRate(ThreadLocalRandom.current().nextDouble(learningRateRange[0], learningRateRange[1]));

        // Input layer is our 'previous' layer
        NeuronLayer previousLayer = new NeuronLayer(
                ThreadLocalRandom.current().nextInt(1, maxNeuronsPerLayer),
                numInputs
        );

        net.addLayer(previousLayer);

        for (int i=0; i < numHiddenLayers; i++) {

            // Make a new layer that is compatible with our last one
            // which means the new layer's inputs per neuron has to match up
            // with the last layer's neuron count because it's fully connected

            NeuronLayer layer = new NeuronLayer(
                    ThreadLocalRandom.current().nextInt(1, maxNeuronsPerLayer),
                    previousLayer.numNeurons
            );

            net.addLayer(layer);

            previousLayer = layer;
        }

        // Finally add the output layer
        net.addLayer(numOutputs, previousLayer.numNeurons);

        return net;
    }
}
