package avalanche.neuralnet.util.activation;

import avalanche.num.Expression;
import avalanche.num.util.MathUtils;

public class ELU implements Expression {
    private double a;
    public ELU(double a) {
        this.a = a;
    }
    public double getA() {
        return a;
    }
    public double evaluate(double input) {
        return input < 0 ? a * (Math.pow(MathUtils.e, input) - 1) : input;
    }
}