package avalanche.neuralnet.nets;

import avalanche.neuralnet.util.Activation;
import avalanche.neuralnet.util.NeuronLayer;
import avalanche.num.Expression;
import avalanche.num.Matrix;

import java.io.*;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ThreadLocalRandom;

import static avalanche.neuralnet.util.Activation.EXPONENTIAL_LINEAR_UNITS;

/**
 * @version 1.4
 * Here be the serialization
 *
 * TODO: Different activation funcs for each layer
 */
public class NeuralNet implements Serializable {

	private final double e = 2.7183;
	private final double a = 1;			// Hyperparameter for Exponential Linear Units
	protected double learningRate = 1;

	/**
	 * layers is protected because that would make it simpler for {@link GeneticAlgorithm}
	 */
	protected List<NeuronLayer> layers;
	protected Activation activation;

	// Initialize expressions to avoid making new ones
	// every time they're called. Protected not private in order
	// to simplify GeneticAlgorithm
	private Expression sigmoidExpression   = new Sigmoid();
	private Expression tanhExpression      = new TanH();
	private Expression eluExpression       = new ELU();
	private Expression reluExpression      = new RELU();
	// Derivatives
	private Expression sigDerivExpression  = new SigmoidDerivative();
	private Expression tanhDerivExpression = new TanHDerivative();
	private Expression eluDerivExpression  = new ELUDerivative();
	private Expression reluDerivExpression = new RELUDerivative();

	// Public constructors
	public NeuralNet() {
		// Overloaded constructor, with no layers and ELU Activation
		this(new ArrayList<>(), EXPONENTIAL_LINEAR_UNITS);
	}
	public NeuralNet(Activation activation) {
		this(new ArrayList<>(), activation);
	}
	public NeuralNet(List<NeuronLayer> startingLayers, Activation activation) {
		layers = startingLayers;
		this.activation = activation;
	}

	// Private classes for sigmoid, elu and deriv expressions
	// Activation functions
	private class Sigmoid implements Expression {
		public double evaluate(double input) {
			return 1/(1+Math.pow(e, -input));
		}
	}
	private class TanH implements Expression {
		public double evaluate(double input) {
			return Math.tanh(input);
		}
	}
	private class ELU implements Expression {
		public double evaluate(double input) {
			return input < 0 ? a * (Math.pow(e, input) - 1) : input;
		}
	}
	private class RELU implements Expression {
		public double evaluate(double input) {
			return input < 0 ? 0 : input;
		}
	}
	// Derivatives
	private class SigmoidDerivative implements Expression {
		public double evaluate(double input) {
			return input * (1 - input);
		}
	}
	private class TanHDerivative implements Expression {
		public double evaluate(double input) {
			// Math.tanh(input) or just input?
			return 1 - (Math.tanh(input) * Math.tanh(input));		// Because this is a simple square, don't use math.pow
		}
	}
	private class ELUDerivative implements Expression {
		public double evaluate(double input) {
			return input < 0 ? input + a : 1;
		}
	}
	private class RELUDerivative implements Expression {
		public double evaluate(double input) {
			return input <= 0 ? 0 : 1;
		}
	}
	@Deprecated
	private class Clamp implements Expression {

		int[] cutoff;

		Clamp(int[] cutoffRange) {
			cutoff = cutoffRange;
		}

		public double evaluate(double input) {
			// Clamp the value between the cutoff ranges
			if (input < cutoff[0]) return cutoff[0];
			if (input > cutoff[1]) return cutoff[1];
			return input;
		}
	}

	// Other methods
	// Setters
	public void addLayer(int neuronCount, int inputsPerNeuron) {
		layers.add(new NeuronLayer(neuronCount, inputsPerNeuron));
	}
	public void addLayer(NeuronLayer newLayer) {
		layers.add(newLayer);
	}
	public void addLayer(NeuronLayer newLayer, int index) {
		layers.add(index, newLayer);
	}
	public NeuronLayer removeLayer(int index) {
		return layers.remove(index);
	}
	public void setLearningRate(double newLearningRate) {
		learningRate = newLearningRate;
	}
	
	// Private methods
	private Matrix sigmoid(Matrix matrix) {
		return matrix.useExpression(sigmoidExpression);
	}
	private Matrix tanh(Matrix matrix) { return matrix.useExpression(tanhExpression); }
	private Matrix elu(Matrix matrix) {
		return matrix.useExpression(eluExpression);
	}
	private Matrix relu(Matrix matrix) { return matrix.useExpression(reluExpression); }
	// Derivatives
	private Matrix sigmoidDerivative(Matrix matrix) {
		return matrix.useExpression(sigDerivExpression);
	}
	private Matrix tanhDerivative(Matrix matrix) { return matrix.useExpression(tanhDerivExpression); }
	private Matrix eluDerivative(Matrix matrix) {
		return matrix.useExpression(eluDerivExpression);
	}
	private Matrix reluDerivative(Matrix matrix) { return matrix.useExpression(reluDerivExpression); }

	// Method to generate a dropout matrix
	private Matrix generateDropoutMatrix(Matrix matrix, double dropoutPercentage) {
		Matrix dropoutMatrix = Matrix.fillEmpty(matrix.numRows(), matrix.numCols(), 1);
		// Make some of the values 0 for 'dropout'. Also compensate the non-dropout values
		dropoutMatrix.useExpression(
				value -> ThreadLocalRandom.current().nextDouble() < dropoutPercentage ? 0 : value
		).scalarProduct(1/(1-dropoutPercentage));
		return dropoutMatrix;
	}

	private Matrix[] layeredThink(Matrix inputs) throws Exception {
		return layeredThink(inputs, false);
	}
	private Matrix[] layeredThink(Matrix inputs, boolean dropout) {
		
		Matrix[] outputs = new Matrix[layers.size()];
		Matrix previousOutput = inputs;
		
		for (int index=0; index<outputs.length; index++) {
			NeuronLayer layer = layers.get(index);
			
			// Dot multiply with synaptic weights, also apply our activation function
			Matrix output = previousOutput.dotProduct(layer.synapticWeights);

			// Dropout to the hidden layers + input layer
			if (dropout && index < outputs.length - 1) {
				output = output.multiply(generateDropoutMatrix(output, 0.25));
			}

			// TODO: Actually understand what's happening right here
			switch (activation) {
				case EXPONENTIAL_LINEAR_UNITS:
					output = elu(output);
					break;
				case RECTIFIED_LINEAR_UNITS:
					output = relu(output);
					break;
				case TANH:
					output = tanh(output);
					break;
				default:        // Sigmoid
					output = sigmoid(output);
					break;
			}

			outputs[index] = output;
			
			previousOutput = output;

		}

		return outputs;
	}

	// Think method that only returns result from final layer
	// Clamp between clamp range, overloaded
	public Matrix think(Matrix input) throws Exception{
		// No clamping
		return think(input, null);
	}
	public Matrix think(Matrix input, @Deprecated int[] clampRange) throws Exception{

		Matrix[] thinkResult = layeredThink(input, false);		// No dropout for final thinkage
		Matrix finalResult = thinkResult[thinkResult.length-1];

		// Clamp if there is a clamp range
		if (clampRange == null) {
			return finalResult;
		} else {
			System.out.println("Clamping");
			return finalResult.useExpression(new Clamp(clampRange));
		}
	}

	// Training method
	public void train(Matrix trainInputs, Matrix trainOutputs, int iterations) {

		for (int iteration=0; iteration<=iterations; iteration++) {

			// Make our layers think
			// Then backpropagate through all of them
			Matrix[] layersOutputs = layeredThink(trainInputs, true);	// Use dropout
			Matrix[] calculatedDeltas = new Matrix[layers.size()];

			boolean visitedLastLayer = false;
			double  compoundedError  = 0;

			// Iterate through our neural layers backwards
			for (int index=layers.size()-1; index>=0; index--) {

				Matrix errorMatrix, matrixDelta;

				// Checks if this is the output layer
				if (!visitedLastLayer) {

					errorMatrix = trainOutputs.sub(layersOutputs[index]);
					visitedLastLayer = true;

				} else {
					
					// Just for easy access
					Matrix lastDelta = calculatedDeltas[index+1];
					NeuronLayer lastLayer = layers.get(index+1);
					
					errorMatrix = lastDelta.dotProduct(lastLayer.synapticWeights.transpose());
				}

				// Apply the activation function's derivative for finding the delta
				switch (activation) {
					case EXPONENTIAL_LINEAR_UNITS:
						matrixDelta = Matrix.multiply(errorMatrix, eluDerivative(layersOutputs[index]));
						break;

					case RECTIFIED_LINEAR_UNITS:
						matrixDelta = Matrix.multiply(errorMatrix, reluDerivative(layersOutputs[index]));
						break;

					case TANH:
						matrixDelta = Matrix.multiply(errorMatrix, tanhDerivative(layersOutputs[index]));
						break;

					default:                // Sigmoid
						matrixDelta = Matrix.multiply(errorMatrix, sigmoidDerivative(layersOutputs[index]));
						break;
				}

				calculatedDeltas[index] = matrixDelta;

				if (iteration%10000 == 0) compoundedError += errorMatrix.sumUpAbs();
			}

			Matrix previousOutput =  trainInputs;

			for (int index=0; index<layers.size(); index++) {
				NeuronLayer layer = layers.get(index);
				
				Matrix adjustment = Matrix.scalarProduct(learningRate, previousOutput.transpose().dotProduct(calculatedDeltas[index]));
				layer.synapticWeights = layer.synapticWeights.add(adjustment);

				previousOutput = layersOutputs[index];
			}
			
			if (iteration%100 == 0) {
				System.out.println("Finished training " + iteration + " iterations \t" + "Error is " + compoundedError);
				System.out.println();
			}
			
		}
	}

	public void printLayers() {
		printLayers(true);
	}
	public void printLayers(boolean printDetails) {

		System.out.println(
				"Printing out Neural Network (" + layers.size() + " layers deep, learning rate: " + learningRate + ")"
		);

		if (printDetails) {
			for (NeuronLayer layer : layers) {
				System.out.print(" Neuron Layer with "
						+ layer.numInputs
						+ " inputs and "
						+ layer.numNeurons
						+ " neurons/outputs  "
				);
				System.out.println(layer.synapticWeights.toCleanString());
			}
		}
	}

	// Static method used to dump nets and load them
	public static void saveNet(NeuralNet net, String fileName) {

		try {
			FileOutputStream     f = new FileOutputStream(fileName);
			BufferedOutputStream b = new BufferedOutputStream(f);
			ObjectOutputStream   o = new ObjectOutputStream(b);

			o.writeObject(net);

			o.close();
			b.close();
			f.close();

		} catch (IOException e) {
			e.printStackTrace();
		}

	}
	public static NeuralNet loadNet(String fileName) {

		try {
			FileInputStream     f = new FileInputStream(fileName);
			BufferedInputStream b = new BufferedInputStream(f);
			ObjectInputStream   o = new ObjectInputStream(b);

			NeuralNet net = (NeuralNet) o.readObject();

			o.close();
			b.close();
			f.close();

			return net;

		} catch (IOException | ClassNotFoundException e) {
			e.printStackTrace();
		}

		return null;
	}
}
