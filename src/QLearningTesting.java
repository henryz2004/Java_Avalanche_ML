import avalanche.qlearning.QLearningMS;
import avalanche.qlearning.state.EndState;
import avalanche.qlearning.state.State;
import avalanche.qlearning.state.StateGenerator;

import java.util.List;

public class QLearningTesting {
    public static void main(String[] args) throws Exception{

        QLearningMS<Integer> qbrain = new QLearningMS<>(0.8);
        int node0 = qbrain.addState(new State<>(0, 0), false);
        int node1 = qbrain.addState(new State<>(1, 0), false);
        int node2 = qbrain.addState(new State<>(2, 0), false);
        int node3 = qbrain.addState(new State<>(3, 0), false);
        int node4 = qbrain.addState(new State<>(4, 0), false);
        int node5 = qbrain.addState(new State<>(5, 100), true);   // Goal state

        State<Integer> endNode = qbrain.getState(node5);

        qbrain.setEndState(new EState(endNode));

        qbrain.mutualLink(node0, node4);
        qbrain.mutualLink(node1, node3);
        qbrain.mutualLink(node1, node5);
        qbrain.mutualLink(node2, node3);
        qbrain.mutualLink(node3, node4);
        qbrain.mutualLink(node4, node5);

        qbrain.printRewards();
        System.out.println();

        qbrain.printQMatrix();
        System.out.println();

        System.out.println("Training");

        qbrain.train(800, new SGen(), null);

        qbrain.printQMatrix();

        System.out.println(qbrain.qThink(node4, null));
    }

    static class SGen implements StateGenerator {
        public List<State> nextState(State state) {
            // Return what the state's linked to
            return state.linksTo;
        }
    }
    static class EState implements EndState {
        State<Integer> end;
        EState(State<Integer> endNode) {
            end = endNode;
        }

        public boolean isEndState(State state) {
            // If end == end state then yes
            return end.equals(state);
        }
    }
}
