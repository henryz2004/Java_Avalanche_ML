import avalanche.neuralnet.util.conv.imgrep.MultiChannelImage;
import avalanche.num.Matrix;
import avalanche.num.util.MathUtils;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;

public class MultiChannelImageTesting {

    public static void main(String[] args) throws IOException {

        MultiChannelImage mci = new MultiChannelImage(new File("resources/TestImage.png"), false);
        List<Matrix> channels = mci.getChannels();

        for (Matrix matrix : channels) {
            double[][] nestedArray = matrix.toArray();
            for (double[] doubles : nestedArray) {
                double[] rounded = new double[doubles.length];
                for (int i=0; i<rounded.length; i++) {
                    rounded[i] = MathUtils.round(doubles[i], 1);
                }
                System.out.println(Arrays.toString(rounded));
            }
            System.out.println();
        }
    }
}
