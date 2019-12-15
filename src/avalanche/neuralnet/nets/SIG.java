package avalanche.neuralnet.nets;

import avalanche.neuralnet.util.Activation;
import avalanche.neuralnet.util.NeuronLayer;
import avalanche.num.Expression;
import avalanche.num.Matrix;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static avalanche.neuralnet.util.Activation.EXPONENTIAL_LINEAR_UNITS;

@Deprecated
public class SIG {

    private final double e = 2.718281828459045;
    private double learningRate = 1;

    /**
     * layers is protected because that would make it simpler for {@link GeneticAlgorithm}
     */
    protected List<NeuronLayer> layers;
    protected Activation activation;

    // Initialize expressions to avoid making new ones
    // every time they're called. Protected not private in order
    // to simplify GeneticAlgorithm
    protected Expression sigmoidExpression  = new Sigmoid();

    // Private classes for sigmoid, elu and deriv expressions
    private class Sigmoid implements Expression {
        public double evaluate(double input) {
            return 1/(1+Math.pow(e, -input));
        }
    }

    // Public constructors
    public SIG() {
        // Overloaded constructor, with no layers and ELU Activation
        this(new ArrayList<>(), EXPONENTIAL_LINEAR_UNITS);
    }
    public SIG(Activation activation) {
        this(new ArrayList<>(), activation);
    }
    public SIG(List<NeuronLayer> startingLayers, Activation activation) {
        layers = startingLayers;
        this.activation = activation;
    }

    // Other methods
    // Setters
    public void addLayer(NeuronLayer newLayer) {
        layers.add(newLayer);
    }
    public void setLearningRate(double newLearningRate) {
        learningRate = newLearningRate;
    }

    // Private methods
    private Matrix sigmoid(Matrix matrix) {
        return matrix.useExpression(sigmoidExpression);
    }
    private Matrix sigmoidDerivative(Matrix matrix) {
        // The matrix has been sigmoided already, therefore
        // We just multiply itself by a subfromscalar(1)
        try {
            return matrix.multiply(matrix.subFromScalar(1));
        } catch (Exception e) {
            e.printStackTrace();
            return Matrix.identityMatrix(matrix.numRows());
        }
    }

    private Matrix[] layeredThink(Matrix inputs) throws Exception {
        return layeredThink(inputs, false);
    }
    private Matrix[] layeredThink(Matrix inputs, boolean debug) throws Exception{

        Matrix[] outputs = new Matrix[layers.size()];
        Matrix previousOutput = inputs;

        for (int index=0; index<outputs.length; index++) {
            NeuronLayer layer = layers.get(index);

            // Dot multiply with synaptic weights, also apply our activation function
            Matrix output = sigmoid(previousOutput.dotProduct(layer.synapticWeights));

            outputs[index] = output;

            previousOutput = output;

        }
        if (debug) System.out.println(Arrays.toString(outputs));
        return outputs;
    }

    // Think method that only returns result from final layer
    // Clamp between clamp range, overloaded
    public Matrix think(Matrix input) throws Exception{
        // No clamping
        return think(input, null);
    }
    public Matrix think(Matrix input, int[] clampRange) throws Exception{

        Matrix[] thinkResult = layeredThink(input, true);
        Matrix finalResult = thinkResult[thinkResult.length-1];

        return finalResult;
    }

    // Training method
    public void train(Matrix trainInputs, Matrix trainOutputs, int iterations) throws Exception{

        for (int iteration=1; iteration<=iterations; iteration++) {

            // Make our layers think
            // Then backpropagate through all of them
            Matrix[] layersOutputs = layeredThink(trainInputs);

            Matrix[] calculatedDeltas = new Matrix[layers.size()];

            boolean visitedLastLayer = false;

            // Iterate through our neural layers backwards (backpropagation)
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

                matrixDelta = errorMatrix.multiply(sigmoidDerivative(layersOutputs[index]));

                calculatedDeltas[index] = matrixDelta;
            }

            Matrix previousOutput = trainInputs;

            for (int index=0; index<layers.size(); index++) {
                NeuronLayer layer = layers.get(index);

                //Matrix adjustment = Matrix.scalarProduct(learningRate, previousOutput.transpose().dotProduct(calculatedDeltas[index]));
                layer.synapticWeights = layer.synapticWeights.add(previousOutput.transpose().dotProduct(calculatedDeltas[index]));

                previousOutput = layersOutputs[index];
            }

            if (iteration%10000 == 0) System.out.println("Finished training " + iteration + " iterations");

        }
    }
}
