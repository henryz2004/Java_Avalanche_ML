package avalanche.qlearning.state;

import avalanche.qlearning.QLearning;

import java.util.LinkedList;
import java.util.List;

public class State<T> {

    public double reward;
    public T state;

    public List<State<T>> linksTo;
    public List<State<T>> linksFrom;

    public State(T state, double reward) {
        this.state = state;
        this.reward = reward;

        linksTo = new LinkedList<>();
        linksFrom = new LinkedList<>();
    }
    public State(T state, double reward, List<State<T>> linksTo, List<State<T>> linksFrom) {
        this(state, reward);

        this.linksTo = linksTo;
        this.linksFrom = linksFrom;
    }

    public static void linkTogether(QLearning qlearn, int state, int otherState) {
        linkTogether(qlearn.getState(state), qlearn.getState(otherState));
    }
    public static void linkTogether(State state, State otherState) {
        // Links state to otherState
        state.addLinkTo(otherState);
        otherState.addLinkFrom(state);
    }

    public static void mutualLink(QLearning qlearn, int state, int otherState) {
        linkTogether(qlearn, state, otherState);
        linkTogether(qlearn, otherState, state);
    }
    public static void mutualLink(State state, State otherState) {
        linkTogether(state, otherState);
        linkTogether(otherState, state);
    }

    public void addLinkTo(State state) {
        linksTo.add(state);
    }
    public void addLinkFrom(State state) {
        linksFrom.add(state);
    }

    public boolean isLinkedTo(State otherState) {
        return linksTo.contains(otherState);
    }

    @Override
    public boolean equals(Object other) {
        if (!(other instanceof State)) return false;

        // Whether or not two states are equal depends soley on state
        return state.equals(((State<T>)other).state);       // state is expected to have .equals
    }

    @Override
    public String toString() {
        return state.toString();
    }
}
