package avalanche.neuralnet.nets;

import avalanche.neuralnet.util.Activation;
import avalanche.neuralnet.util.NeuronLayer;
import avalanche.neuralnet.util.architecture.NetArchitecture;
import avalanche.neuralnet.util.fitness.FitnessComparator;
import avalanche.neuralnet.util.fitness.FitnessFunction;
import avalanche.neuralnet.util.fitness.FitnessTuple;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.ThreadLocalRandom;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 * Just a collection of static methods that will train an input net
 *
 * @version 0.1
 */
public class GeneticAlgorithm {

    private static FitnessComparator fitnessComparator   = new FitnessComparator(); // One comparator is good enough

    // Cannot be instantiated
    private GeneticAlgorithm() {}

    public static List<NeuralNet> train(NetArchitecture blueprint,
                                  FitnessFunction ff,
                                  double percentRetainFittest,
                                  double percentRetainPoorest,
                                  int populationSize,
                                  int iterations) {

        // Make random population of starting nets
        // Use List<NeuralNet> instead of NeuralNet[] because it might be easier
        List<NeuralNet> population = new ArrayList<>();

        // Generate random nets and add the net to the population
        // Also print out our population
        for (int i=0; i<populationSize; i++) {
            NeuralNet net = blueprint.constructNet();
            net.printLayers();
            population.add(net);
        }

        // Calculate the number of nets to keep
        int numTopNetsRetain = (int) Math.round(percentRetainFittest * populationSize);
        int numBadNetsRetain = (int) Math.round(percentRetainPoorest * populationSize);


        for (int iteration=0; iteration < iterations; iteration++) {

            List<FitnessTuple> netsFitness = new ArrayList<>();

            // Score each member of our population
            for (NeuralNet net : population) {
                netsFitness.add(new FitnessTuple(net, ff.fitness(net)));
            }

            netsFitness.sort(fitnessComparator);       // Sort the fitnesses using our custom comparator

            // Make lists of the surviving nets
            List<FitnessTuple> bestNets  = subList(netsFitness, 0, numTopNetsRetain);

            int startIndex = numTopNetsRetain
                    + ThreadLocalRandom.current().nextInt(
                            populationSize
                                    - numTopNetsRetain
                                    - numBadNetsRetain);
            int endIndex = startIndex + numBadNetsRetain;
            List<FitnessTuple> otherNets = subList(netsFitness, startIndex, endIndex);

            // Kill the rest of the nets
            population.clear();

            // If the number of surviving nets is odd, then we must kill an extra survivor
            if ((numTopNetsRetain + numBadNetsRetain) % 2 == 1) {
                otherNets.remove(ThreadLocalRandom.current().nextInt(numBadNetsRetain));
            }

            // Combine all the survivors into one list for iteration and breeding
            List<FitnessTuple> survivors = Stream.of(bestNets, otherNets)
                                                 .flatMap(Collection::stream)
                                                 .collect(Collectors.toList());

            // Breed pairs of neural nets and continue doing so until we have a brand new population
            while (population.size() < populationSize) {

                for (int i=0; i<survivors.size(); i+=2) {
                    if (population.size() >= populationSize) break;

                    NeuralNet mother = survivors.get(i).net;
                    NeuralNet father = survivors.get(i).net;

                    population.add(breed(mother, father));

                }

                Collections.shuffle(survivors);     // We must shuffle the survivors in order to maintain genetic diversity
            }

            if (iteration % 10 == 0) System.out.println("Done with " + iteration + " generations");
        }

        // Return the population
        return population;
    }

    // Genetic helper functions
    private static NeuralNet breed(NeuralNet mother, NeuralNet father) {

        NeuralNet child = new NeuralNet(Activation.SIGMOID);

        // Pick random characteristics to copy
        // Don't use setters/getters because this seems a tad bit more clean
        child.learningRate = ThreadLocalRandom.current().nextBoolean() ? mother.learningRate : father.learningRate;

        // Due to the fact that this evolves fixed-topology nets, we can swap layers without worrying about
        // incompatibility
        for (int i=0; i<mother.layers.size(); i++) {
            child.layers.add(ThreadLocalRandom.current().nextBoolean() ? mother.layers.get(i) : father.layers.get(i));
        }

        // Mutate
        // TODO Add mutation where a layer is added/removed or neurons are added/removed
        // TODO Add user control for which mutations to mutate

        // Weight mutations
        // https://stackoverflow.com/questions/31708478/how-to-evolve-weights-of-a-neural-network-in-neuroevolution
        for (NeuronLayer layer : child.layers) {

            switch (ThreadLocalRandom.current().nextInt(5)) {

                // Completely replace parts of a layer with random weights
                case 0:
                    layer.synapticWeights = layer.synapticWeights.useExpression(
                            value -> ThreadLocalRandom.current().nextInt(10) == 0
                                    ? ThreadLocalRandom.current().nextDouble(-2, 2)
                                    : value);   // 10% chance of weight change

                    // Multiply by number
                case 1:
                    layer.synapticWeights = layer.synapticWeights.useExpression(
                            value -> value * ThreadLocalRandom.current().nextDouble(0.5, 1.5)
                    );

                    // Add/subtract number
                case 2:
                    layer.synapticWeights = layer.synapticWeights.scalarSum(ThreadLocalRandom.current().nextDouble(-1, 1));

                    // Flip signs
                case 3:
                    layer.synapticWeights = layer.synapticWeights.useExpression(
                            value -> value * (ThreadLocalRandom.current().nextInt(10) == 0 ? -1 : 1)
                    );  // 10% chance of weight flip

                default:
                    // DO NOTHING
            }
        }

        // Change the learning rate, not really necessary
        child.learningRate += ThreadLocalRandom.current().nextDouble(-0.05, 0.05);
        if (child.learningRate <= 0) child.learningRate = 0.1;

        return child;
    }

    // Utility helper functions
    private static <E> List<E> subList(List<E> originalList, int start, int end) {
        // The thing about this version is that this doesn't return a view; it returns
        // a brand new list that contains the objects in that range
        List<E> sub = new ArrayList<>();
        for (int i=start; i<end; i++) {
            sub.add(originalList.get(i));
        }

        return sub;
    }
}
