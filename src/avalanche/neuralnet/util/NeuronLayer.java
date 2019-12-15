package avalanche.neuralnet.util;

import avalanche.num.Matrix;

import java.io.Serializable;

public class NeuronLayer implements Serializable {

	/**
	 * The synaptic weight matrix will have a structure similar to:
	 *
	 * 			Neuron Count
	 * Inputs   x    x    x
	 *  per     x    x    x
	 * Neuron   x    x    x
	 */

	public int numNeurons;		// Cols
	public int numInputs;		// Rows
	public Matrix synapticWeights;
	
	public NeuronLayer(int neuronCount, int inputsPerNeuron) {
		// Initialize our weights to be random
		// Avoid dead nets by making it impossible for the nets to start out with weights
		// too close to 0
		// TODO: Add matrix function to initialize a random matrix with a seed without range1 and 2
		synapticWeights = Matrix.fillRandom(inputsPerNeuron, neuronCount, new double[] {-1, 0}, new double[] {0, 1});

		numNeurons = neuronCount;
		numInputs  = inputsPerNeuron;
	}
	public NeuronLayer(int neuronCount, int inputsPerNeuron, long seed) {
		synapticWeights = Matrix.fillRandom(inputsPerNeuron, neuronCount, new double[] {-1, -0}, new double[] {0, 1}, seed);

		numNeurons = neuronCount;
		numInputs  = inputsPerNeuron;
	}

	@Override
	public String toString() {
		return synapticWeights.toString();
	}
}
