package avalanche.qlearning.state;

import java.util.List;

public interface StateGenerator<T> {
    // Generates a list of next states given state
    List<State<T>> nextState(State<T> state);
}
