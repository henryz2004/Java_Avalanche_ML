import avalanche.neuralnet.nets.ConvolutionalNeuralNet;
import avalanche.neuralnet.util.conv.imgrep.MultiChannelImage;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;

public class CNNTesting4 {

    public static void main(String[] args) throws IOException {

        ConvolutionalNeuralNet loadedCNN = ConvolutionalNeuralNet.loadNet("ConvolutionalNeuralNet3.ava");

        // Test the cnn
        MultiChannelImage o = new MultiChannelImage(new File("D:\\eclipse-workspace\\Machine_Learning\\_images\\_testing images\\o.png"), true);
        MultiChannelImage r = new MultiChannelImage(new File("D:\\eclipse-workspace\\Machine_Learning\\_images\\_testing images\\r.png"), true);
        MultiChannelImage t = new MultiChannelImage(new File("D:\\eclipse-workspace\\Machine_Learning\\_images\\_testing images\\t.png"), true);

        // Just to see if it can do ANYTHING
        MultiChannelImage r2 = new MultiChannelImage(new File("D:\\eclipse-workspace\\Machine_Learning\\_images\\_training images\\rectangle\\0.png"), true);
        MultiChannelImage blank = new MultiChannelImage(new File("D:\\eclipse-workspace\\Machine_Learning\\_images\\_testing images\\blank.png"), true);

        System.out.println(Arrays.toString(loadedCNN.classify(o)));
        System.out.println(Arrays.toString(loadedCNN.classify(r)));
        System.out.println(Arrays.toString(loadedCNN.classify(t)));
        System.out.println(Arrays.toString(loadedCNN.classify(r2)));
        System.out.println(Arrays.toString(loadedCNN.classify(blank)));     // Disturbing.... [0,1,0]?
    }
}
