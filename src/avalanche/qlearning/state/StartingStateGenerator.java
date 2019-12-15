package avalanche.qlearning.state;

public interface StartingStateGenerator<T> {
    // Generates starting states for q-learning
    State<T> generateStartingState();
}
