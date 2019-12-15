package avalanche.neuralnet.util.fitness;

import avalanche.neuralnet.nets.NeuralNet;

public class FitnessTuple implements Comparable<FitnessTuple> {

    public NeuralNet net;
    public double    fitness;

    public FitnessTuple(NeuralNet neuralNet, double netFitness) {
        net     = neuralNet;
        fitness = netFitness;
    }

    @Override
    public int compareTo(FitnessTuple otherTuple) {
        if (fitness > otherTuple.fitness) {
            return 1;
        } else if (fitness < otherTuple.fitness) {
            return -1;
        } else {
            return 0;
        }
    }
}
