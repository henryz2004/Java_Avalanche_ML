package avalanche.qlearning;

import avalanche.qlearning.state.EndState;
import avalanche.qlearning.state.StartingStateGenerator;
import avalanche.qlearning.state.State;
import avalanche.qlearning.state.StateGenerator;
import avalanche.qlearning.storage.MassStore;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ThreadLocalRandom;

/**
 * This is the updated version of the deprecated {@link avalanche.qlearning.QLearning} class
 * All code that had previously used the deprecated class should still work with this class.
 * @version 1.2  8/2/17
 */
public class QLearningMS<T> {
    // Generic because the states can be effectively, anything
    // This'll be more challenging, considering the fact that there
    // May be no definite end point

    // Now using MassStore-age!
    private MassStore QMatrix;
    private MassStore RMatrix;

    private List<State<T>> states;
    private EndState endState;

    private double discountFactor;

    // Used for timing
    private long start;

    // Public constructor
    public QLearningMS(double gamma) {
        QMatrix   = new MassStore(0);
        RMatrix   = new MassStore(-1);
        states    = new ArrayList<>();
        discountFactor = gamma;
        start     = System.currentTimeMillis();         // Logs when the class was made
    }
    public QLearningMS(double gamma, EndState endState) {
        this(gamma);
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
        if (RMatrix.get(startIndex, startIndex/*randomStateIndex*/) < 0) {
            RMatrix.set(startIndex, startIndex/*randomStateIndex*/, (double) 0);
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
                QValue = QMatrix.get(stateIndex, states.indexOf(nextState));
            }
            if (QValue > maxQ) {
                maxQ = QValue;
            }
        }

        return maxQ;
    }

    /**
     * @version 1.2
     *  - Overloaded
     *  - Toggle normalization
     */
    public void train(int numEpisodes, StateGenerator<T> sgen, StartingStateGenerator<T> ssgen) throws Exception {
        train(numEpisodes, sgen, ssgen, true);      // Normalize.
    }
    public void train(int numEpisodes, StateGenerator<T> sgen, StartingStateGenerator<T> ssgen, boolean normalize) throws Exception {
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
                QMatrix.set(currStateIndex,
                        randNextStateIndex,
                        RMatrix.get(currStateIndex, randNextStateIndex) + discountFactor * maxQofState(randNextStateIndex, sgen)
                );

                currState = randNextState;
                currStateIndex = randNextStateIndex;

            } while (!endState.isEndState(currState));
        }

        if (normalize) {
            // Finished training, now normalize the values between 0 and 1

            // Normalization
            //logTime();
            QMatrix.normalize();
            //printTime();
        }
    }

    // The thinker, overloaded + helper functions
    private int maxActionOfState(int QRowIndex) {
        // Honestly such a waste of a few lines. Very similar to maxQofState

        int actionIndex = 0;
        double maxQ = Double.NEGATIVE_INFINITY;

        for (int index=0; index<QMatrix.numCols(); index++) {
            if (QMatrix.get(QRowIndex, index) > maxQ) {
                actionIndex = index;
                maxQ = QMatrix.get(QRowIndex, index);
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
     * O(n**2), use the {@link this#linkFrom(int, int)} or {@link this#mutualLink(int, int)}
     */
    @Deprecated
    public void updateRewards() {

        RMatrix.clear();

        for (State<T> state : states) {

            RMatrix.addRow();
            int rowIndex = RMatrix.numRows()-1;

            // If this state is linked to any of the other states, use their reward
            // Otherwise, use -1
            for (State<T> compState : states) {
                if (state.isLinkedTo(compState)) {
                    QMatrix.addToRow(rowIndex, compState.reward);
                } else {
                    QMatrix.addToRow(rowIndex, (double) -1);
                }
            }
        }
    }

    public void linkFrom(int state, int otherState) {
        State.linkTogether(states.get(state), states.get(otherState));

        // Update reward
        RMatrix.set(state, otherState, states.get(otherState).reward);
    }
    public void mutualLink(int state, int otherState) {
        State.mutualLink(states.get(state), states.get(otherState));

        // Update rewards
        RMatrix.set(state, otherState, states.get(otherState).reward);
        RMatrix.set(otherState, state, states.get(state).reward);
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
        QMatrix.addCol();
        RMatrix.addCol();

        // Add the new row state
        List<Double> rewardRow = new ArrayList<>();
        for (int i=0; i<=stateIndex; i++) {
            rewardRow.add((double) -1);

            if (i==stateIndex && linkSelf) {
                state.addLinkTo(state);
                state.addLinkFrom(state);
                rewardRow.set(i, state.reward);
            }
        }

        QMatrix.addRow();
        RMatrix.addRowList(rewardRow);

        // Link the link froms and link tos
        for (int index : linksFrom) {
            state.addLinkFrom(states.get(index));

            // Also update the reward matrix
            RMatrix.set(index, stateIndex, state.reward);
        }
        for (int index : linksTo) {
            state.addLinkTo(states.get(index));

            // Update reward matrix
            RMatrix.set(stateIndex, index, states.get(index).reward);
        }

        return stateIndex;     // Returns the index of this new state
    }

    public State<T> getState(int index) {
        return states.get(index);
    }
    public List<State<T>> getStates() {
        return states;
    }

    // Timing functions
    private void logTime() {
        start = System.currentTimeMillis();
    }
    private void printTime() {
        // If logTime was not called previously, then this will print time since class creation
        System.out.println(System.currentTimeMillis() - start);
    }

    // Printing functions
    public void printNumStates() {
        System.out.println(states.size());
    }
    public void printRewards() {
        prettyPrint2D(RMatrix);
    }
    public void printQMatrix() {
        prettyPrint2D(QMatrix);
    }
    private  void prettyPrint2D(MassStore eList) {
        for (int row=0; row < eList.numRows(); row++) {
            for (int col=0; col < eList.numCols(); col++) {
                System.out.printf("%10.1f", eList.get(row, col));
            }
            System.out.println();
        }
    }
    private <E> E logged(E toLog, String logName) {
        System.out.println("Logging " + logName + ": " + toLog);
        return toLog;
    }
}
