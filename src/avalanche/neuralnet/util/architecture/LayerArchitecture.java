package avalanche.neuralnet.util.architecture;

import avalanche.neuralnet.util.NeuronLayer;

public class LayerArchitecture {

    public int numNeurons;
    public int numInputs;

    // Constructor is the same as NeuronLayer
    public LayerArchitecture(int neuronCount, int inputsPerNeuron) {
        numNeurons = neuronCount;
        numInputs  = inputsPerNeuron;
    }

    public NeuronLayer constructLayer() {
        return constructLayer(-1);
    }
    public NeuronLayer constructLayer(long seed) {
        if (seed != -1) {
            return new NeuronLayer(numNeurons, numInputs, seed);
        } else {
            return new NeuronLayer(numNeurons, numInputs);
        }
    }
}
