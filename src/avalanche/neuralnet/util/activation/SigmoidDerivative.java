package avalanche.neuralnet.util.activation;

import avalanche.num.Expression;

public class SigmoidDerivative implements Expression {
    public double evaluate(double input) {
        return input * (1 - input);
    }
}