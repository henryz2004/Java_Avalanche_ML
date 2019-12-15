import avalanche.neuralnet.nets.ConvolutionalNeuralNet;
import avalanche.neuralnet.util.conv.imgrep.MultiChannelImage;

import java.io.File;
import java.util.Arrays;

public class CNNTesting2 {

    public static void main(String[] args) throws Exception {

        ConvolutionalNeuralNet loadedCNN = ConvolutionalNeuralNet.loadNet("ConvolutionalNeuralNet2.ava");

        // Load test images
        MultiChannelImage x = new MultiChannelImage(new File("D:\\eclipse-workspace\\Machine_Learning\\_images\\_testing images\\x.png"), true);
        MultiChannelImage x2 = new MultiChannelImage(new File("D:\\eclipse-workspace\\Machine_Learning\\_images\\_testing images\\x2.png"), true);
        MultiChannelImage o = new MultiChannelImage(new File("D:\\eclipse-workspace\\Machine_Learning\\_images\\_testing images\\o.png"), true);
        MultiChannelImage o2 = new MultiChannelImage(new File("D:\\eclipse-workspace\\Machine_Learning\\_images\\_testing images\\o2.png"), true);

        System.out.println(Arrays.toString(loadedCNN.classify(x)));
        System.out.println(Arrays.toString(loadedCNN.classify(x2)));
        System.out.println(Arrays.toString(loadedCNN.classify(o)));
        System.out.println(Arrays.toString(loadedCNN.classify(o2)));
    }
}
