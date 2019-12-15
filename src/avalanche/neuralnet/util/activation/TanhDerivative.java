package avalanche.neuralnet.util.activation;

import avalanche.num.Expression;

public class TanhDerivative implements Expression {
    public double evaluate(double input) {
        // Math.tanh(input) or just input?
        return 1 - input * input;		// Because this is a simple square, don't use math.pow
    }
}