package avalanche.neuralnet.util.activation;

import avalanche.num.Expression;

public class ELUDerivative implements Expression {
    private ELU elu;
    public ELUDerivative(ELU elu) {
        this.elu = elu;
    }
    public double evaluate(double input) {
        return input < 0 ? input + elu.getA() : 1;
    }
}