package avalanche.neuralnet.util.activation;

import avalanche.num.Expression;

public class ReLUDerivative implements Expression {
    public double evaluate(double input) {
        return input <= 0 ? 0 : 1;
    }
}