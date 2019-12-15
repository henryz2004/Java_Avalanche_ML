package avalanche.neuralnet.util.fitness;

import avalanche.neuralnet.nets.NeuralNet;

public interface FitnessFunction {
    // Self-explanatory. Used in GeneticAlgorithm.java

    double fitness(NeuralNet net);
}
