package avalanche.qlearning;

import avalanche.qlearning.state.EndState;
import avalanche.qlearning.state.StartingStateGenerator;
import avalanche.qlearning.state.State;
import avalanche.qlearning.state.StateGenerator;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ThreadLocalRandom;

/**
 * @version 1.0
 * @deprecated
 * This has an obsolete Matrix storage algorithm. Use {@link avalanche.qlearning.QLearningMS} instead.
 * All code that had used this class should work fine with the new one.
 */
@Deprecated
public class QLearning<T> {
    // Generic because the states can be effectively, anything
    // This'll be more challenging, considering the fact that there
    // May be no definite end point

    // Do not use numava matrices because the containers must be dynamic
    // And this doesn't only store doubles
    private List<List<Double>> QMatrix;
    private List<List<Double>> RMatrix;

    private List<State<T>> states;
    private EndState endState;

    private double discountFactor;

    // Public constructor
    public QLearning(double gamma) {
        QMatrix   = new ArrayList<>();
        RMatrix   = new ArrayList<>();
        states    = new ArrayList<>();
        discountFactor = gamma;
    }
    public QLearning(double gamma, int initialSize) {
        QMatrix   = new ArrayList<>(initialSize);
        RMatrix   = new ArrayList<>(initialSize);
        states    = new ArrayList<>(initialSize);
        discountFactor = gamma;
    }
    public QLearning(double gamma, EndState endState) {
        this(gamma);
        this.endState = endState;
    }
    public QLearning(double gamma, int initialSize, EndState endState) {
        this(gamma, initialSize);
        this.endState = endState;
    }

    // The beast is underneath.
    // The training function, preceded by some private helper functions
    // For more information about the 60 line rewrite, go to:
    //     https://docs.google.com/document/d/1tAmydt6xXoT12trtgN0pPluvyewWTwgQvB6L8Eywt6Y/edit?usp=sharing
    private int generateStartingState(StartingStateGenerator<T> ssgen) {

        State<T> startingState = ssgen.generateStartingState();

        if (!states.contains(startingState)) {
            // startingState is foreign, add it to our collection
            return logged(addState(startingState, false), "new state");

        } else {
            return logged(states.indexOf(startingState), "old state");
        }
    }
    private int generateRandomNextState(int startIndex,  StateGenerator<T> sgen) {

        List<State<T>> nextStates = sgen.nextState(states.get(startIndex));

        int randomStateIndex = ThreadLocalRandom.current().nextInt(nextStates.size());
        State<T> randomState = nextStates.get(randomStateIndex);

        if (!states.contains(randomState)) {
            addState(randomState, new int[] {startIndex}, new int[0], false);

            if (!states.get(startIndex).linksTo.contains(randomState)) {
                states.get(startIndex).linksTo.add(randomState);
            }
        }

        // Update R-Matrix if needed
        // < 0 because -1 might? be something odd like -0.999999998 or something
        if (RMatrix.get(startIndex).get(startIndex/*randomStateIndex*/) < 0) {
            RMatrix.get(startIndex).set(startIndex/*randomStateIndex*/, (double) 0);
        }

        return states.indexOf(randomState);     // Return the index of the random state in STATES, not the linksTo
    }
    private double maxQofState(int stateIndex, StateGenerator<T> sgen) {

        List<State<T>> nextStates = sgen.nextState(states.get(stateIndex));

        double maxQ = Double.NEGATIVE_INFINITY;
        for (State<T> nextState : nextStates) {

            double QValue;

            // If this next state is uncharted, make the Q value 0
            if (!states.contains(nextState)) {
                QValue = 0;
            } else {
                QValue = QMatrix.get(stateIndex).get(states.indexOf(nextState));
            }
            if (QValue > maxQ) {
                maxQ = QValue;
            }
        }

        return maxQ;
    }

    public void train(int numEpisodes, StateGenerator<T> sgen, StartingStateGenerator<T> ssgen) throws Exception {
        // TODO: Add ability to train until convergence (without being provided a constant numEpisodes amount

        // First check if there is an end state to aim for
        if (endState == null) {
            throw new Exception("End state must be initialized before training. Use .setEndState(endState).");
        }

        for (int iteration=0; iteration<numEpisodes; iteration++) {
            System.out.println("Training iteration number " + iteration);

            int currStateIndex;

            // If there's no StartingStateGenerator then pick one from preexisting states
            // Otherwise, use a helper function and generate a starting index
            if (ssgen == null) {
                currStateIndex = ThreadLocalRandom.current().nextInt(states.size());
            } else {
                currStateIndex = generateStartingState(ssgen);
            }

            State<T> currState;


            do {
                System.out.println("Iteration " + iteration + ", exploring. Total state count: " + states.size());

                int randNextStateIndex = generateRandomNextState(currStateIndex, sgen);
                State<T> randNextState = states.get(randNextStateIndex);

                // Update QMatrix
                QMatrix.get(currStateIndex).set(
                        randNextStateIndex,
                        RMatrix.get(currStateIndex).get(randNextStateIndex) + discountFactor * maxQofState(randNextStateIndex, sgen)
                );

                currState = randNextState;
                currStateIndex = randNextStateIndex;

            } while (!endState.isEndState(currState));
        }

        // Finished training, now normalize the values between 0 and 1
        // First find the max Q
        double maxQ = Double.NEGATIVE_INFINITY;
        for (List<Double> row : QMatrix) {
            for (Double element : row) {
                if (element > maxQ) maxQ = element;
            }
        }

        // Now divide all the values in the Q Matrix by maxQ (if it isn't 0)
        if (maxQ == 0) {
            throw new Exception("Empty Q-Matrix. Cannot finish training. Make sure numEpisodes > 0");
        }

        for (List<Double> row : QMatrix) {
            for (int i=0; i<row.size(); i++) {
                row.set(i, row.get(i)/maxQ);
            }
        }
    }

    // The thinker, overloaded + helper functions
    private int maxActionOfState(int QRowIndex) {
        // Honestly such a waste of a few lines. Very similar to maxQofState

        List<Double> QRow = QMatrix.get(QRowIndex);

        int actionIndex = 0;
        double maxQ = Double.NEGATIVE_INFINITY;

        for (int index=0; index<QRow.size(); index++) {
            if (QRow.get(index) > maxQ) {
                actionIndex = index;
                maxQ = QRow.get(index);
            }
        }

        return actionIndex;
    }
    public List<State<T>> qThink(int stateIndex, StateGenerator<T> sgen) {
        return qThink(states.get(stateIndex), sgen);
    }
    public List<State<T>> qThink(State<T> state, StateGenerator<T> sgen) {
        // The state generator can be null. If it is null, then that means that
        // State<T> state and next descendant states will ALWAYS be mapped, or
        // Already encountered in training.
        //
        // If state generator is not null, then it will be used to generate the
        // Next states, given the current state. If all of the generated states
        // Are uncharted, then pick the one with the most reward.
        // If only some are uncharted, then pick a state from the charted ones.

        // TODO: Add maximum loop iterations (optional, probably)

        List<State<T>> statePath = new ArrayList<>();  // Will contain the list of states to visit, in that order

        int currStateIndex = states.indexOf(state);
        State<T> currState = state;

        if (sgen == null) {

            do {
                // Get the QRow and find the best next move to make
                int bestMoveIndex = maxActionOfState(currStateIndex);

                // Make the best move
                // Add the currState, not the best move, because that'll get added NEXT iteration
                statePath.add(currState);

                currStateIndex = bestMoveIndex;
                currState = states.get(bestMoveIndex);

            } while(!endState.isEndState(currState));

        } else {

            do {
                List<State<T>> nextStates = sgen.nextState(currState);

                // Make a list of all the Q-states in nextStates
                List<State<T>> qStates = new ArrayList<>();
                for (State<T> nextState : nextStates) {
                    if (states.contains(nextState)) qStates.add(nextState);
                }

                // In case of empty qStates, log a 'warning' and pick the one with the highest reward
                if (qStates.size() == 0) {
                    System.out.println("Completely foreign state encountered.");

                    int highestRewardIndex = 0;
                    double highestReward   = Double.NEGATIVE_INFINITY;

                    for (int index=0; index<nextStates.size(); index++) {
                        if (nextStates.get(index).reward > highestReward) {
                            highestRewardIndex = index;
                            highestReward = nextStates.get(index).reward;
                        }
                    }

                    statePath.add(currState);

                    currStateIndex = highestRewardIndex;
                    currState = nextStates.get(highestRewardIndex);

                } else {
                    // Simply find the best Q action and play it
                    // Clone of if (sgen == null) {...}
                    int bestMoveIndex = maxActionOfState(currStateIndex);

                    // Make the best move
                    // Add the currState, not the best move, because that'll get added NEXT iteration
                    statePath.add(currState);

                    currStateIndex = bestMoveIndex;
                    currState = states.get(bestMoveIndex);
                }

            } while (!endState.isEndState(currState));
        }

        statePath.add(currState);           // Finish by adding the endState

        return statePath;
    }

    /**
     * @deprecated
     * O(n**2), use {@link #linkFrom(int, int)} or {@link #mutualLink(int, int)} instead.
     */
    @Deprecated
    public void updateRewards() {

        RMatrix.clear();

        for (State<T> state : states) {

            List<Double> rowList= new ArrayList<>();
            RMatrix.add(rowList);

            // If this state is linked to any of the other states, use their reward
            // Otherwise, use -1
            for (State<T> compState : states) {
                if (state.isLinkedTo(compState)) {
                    rowList.add(compState.reward);
                } else {
                    rowList.add((double) -1);
                }
            }
        }
    }

    public void linkFrom(int state, int otherState) {
        State.linkTogether(states.get(state), states.get(otherState));

        // Update reward
        RMatrix.get(state).set(otherState, states.get(otherState).reward);
    }
    public void mutualLink(int state, int otherState) {
        State.mutualLink(states.get(state), states.get(otherState));

        // Update rewards
        RMatrix.get(state).set(otherState, states.get(otherState).reward);
        RMatrix.get(otherState).set(state, states.get(state).reward);
    }

    // Setters
    public void setEndState(EndState endState) {
        this.endState = endState;
    }
    public int addState(State<T> state) {
        return addState(state, true);
    }
    public int addState(State<T> state, boolean linkSelf) {
        return addState(state, new int[0], new int[0], linkSelf);
    }
    public int addState(State<T> state, int[] linksFrom, int[] linksTo, boolean linkSelf) {
        // Returns the index of this new state for future linkage
        // Increase the size of Q and R
        // linksFrom and linksTo are arrays of indices that link to and from
        // This new state

        int stateIndex = states.size();
        states.add(state);

        // Enlarge our reward matrix and brain matrix
        for (List<Double> row : QMatrix) { row.add((double) 0);  }
        for (List<Double> row : RMatrix) { row.add((double) -1); }

        // Add the new row state
        ArrayList<Double> rewardRow = new ArrayList<>();
        for (int i=0; i<=stateIndex; i++) {
            rewardRow.add((double) -1);

            if (i==stateIndex && linkSelf) {
                state.addLinkTo(state);
                state.addLinkFrom(state);
                rewardRow.set(i, state.reward);
            }
        }

        ArrayList<Double> newQRow = new ArrayList<>();
        QMatrix.add(newQRow);

        for (int i=0; i<QMatrix.size(); i++) {
            newQRow.add((double) 0);
        }

        RMatrix.add(rewardRow);

        // Link the link froms and link tos
        for (int index : linksFrom) {
            state.addLinkFrom(states.get(index));

            // Also update the reward matrix
            RMatrix.get(index).set(stateIndex, state.reward);
        }
        for (int index : linksTo) {
            state.addLinkTo(states.get(index));

            // Update reward matrix
            rewardRow.set(index, states.get(index).reward);
        }

        return stateIndex;     // Returns the index of this new state
    }

    public State<T> getState(int index) {
        return states.get(index);
    }
    public List<State<T>> getStates() {
        return states;
    }

    public void printNumStates() {
        System.out.println(states.size());
    }
    public void printRewards() {
        prettyPrint2D(RMatrix);
    }
    public void printQMatrix() {
        prettyPrint2D(QMatrix);
    }
    private <E> void prettyPrint2D(List<List<E>> eList) {
        for (List<E> row   : eList) {
            for (E element : row) {
                System.out.printf("%10.1f", element);
            }
            System.out.println();
        }
    }
    private <E> E logged(E toLog, String logName) {
        System.out.println("Logging " + logName + ": " + toLog);
        return toLog;
    }
}
