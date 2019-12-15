package avalanche.util.annotations;

public @interface Dangerous {
    String value() default "Do NOT use this dangerous class/method.";
}
