import avalanche.neuralnet.util.conv.dimred.ConvolutionLayer;
import avalanche.neuralnet.util.conv.imgrep.MultiChannelImage;
import avalanche.num.Matrix;
import avalanche.num.util.MathUtils;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import static avalanche.util.ArrayUtils.a;

public class PoolConvTesting {

    public static void main(String[] args) throws IOException {
        // Pooling tests passed

        MultiChannelImage mci = new MultiChannelImage(new File("resources/TestImage.png"), true);

        printImage(mci);

        // Pooling tests
        /*PoolingLayer pool = new PoolingLayer(2, 2, 20);

        printImage(pool.feedSingular(mci));*/

        // Convolution tests

        // New conv layer with 1 filters (3 channels) per input and 1 input
        ConvolutionLayer conv    = new ConvolutionLayer(
                new MultiChannelImage[][] {{new MultiChannelImage(new Matrix[] {
                        Matrix.from2D(new double[][] {
                                a(0D, 0D, 0D),
                                a(0D, 0D, 0D),
                                a(0D, 0D, 0D)
                        }),
                        Matrix.from2D(new double[][] {
                                a(1D, 1D, 1D),
                                a(1D, 1D, 1D),
                                a(1D, 1D, 1D)
                        }),
                        Matrix.from2D(new double[][] {
                                a(0D, 1D, 0D),
                                a(1D, 0D, 1D),
                                a(0D, 1D, 0D)
                        })
                })}}
        );
        MultiChannelImage[][] filters = conv.getFilters();

        for (MultiChannelImage[] inputFilters : filters) {
            for (MultiChannelImage filter : inputFilters) {
                printImage(filter);
            }
        }

        printImage(conv.feedForward(Collections.singletonList(mci)).get(0));
    }

    static void printImage(MultiChannelImage mci) {
        List<Matrix> channels = mci.getChannels();

        System.out.println("Printing image with " + channels.size() + " channels");

        for (int channel=0; channel<channels.size(); channel++) {
            Matrix matrix = channels.get(channel);

            System.out.println("Channel " + channel);

            double[][] nestedArray = matrix.toArray();
            for (double[] doubles : nestedArray) {
                double[] rounded = new double[doubles.length];
                for (int i=0; i<rounded.length; i++) {
                    rounded[i] = MathUtils.round(doubles[i], 5);
                }
                System.out.println(Arrays.toString(rounded));
            }
            System.out.println();
        }
    }
}
