package avalanche.util;

import avalanche.util.annotations.Primitive;

@Primitive
public class StringUtils {

    private StringUtils() {}

    public static String repeat(String repeatingString, int count) {
        StringBuilder sb = new StringBuilder();
        for (int i=0; i<count; i++) {
            sb.append(repeatingString);
        }
        return sb.toString();
    }
}
