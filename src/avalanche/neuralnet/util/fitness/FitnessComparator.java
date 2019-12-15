package avalanche.neuralnet.util.fitness;

import java.util.Comparator;

public class FitnessComparator implements Comparator<FitnessTuple> {
    @Override
    public int compare(FitnessTuple tuple1, FitnessTuple tuple2) {
        return tuple2.compareTo(tuple1);
    }
}
