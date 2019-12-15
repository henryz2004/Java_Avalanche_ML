package avalanche.neuralnet.util.activation;

import avalanche.num.Expression;

public class Tanh implements Expression {
    public double evaluate(double input) {
        return Math.tanh(input);
    }
}