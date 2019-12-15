package avalanche.qlearning.state;

public interface EndState {
    // Interface to determine if the current state is the goal state for q-learning
    boolean isEndState(State currentState);
}
