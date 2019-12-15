package avalanche.neuralnet.util.activation;

import avalanche.num.Expression;
import avalanche.num.util.MathUtils;

public class Sigmoid implements Expression {
    public double evaluate(double input) {
        return 1/(1+Math.pow(MathUtils.e, -input));
    }
}