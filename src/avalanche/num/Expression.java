package avalanche.num;

import java.io.Serializable;

public interface Expression extends Serializable {
	// A way for expressions to be passed into methods
	// Call the evaluate() method
	
	double evaluate(double input);
	
}
