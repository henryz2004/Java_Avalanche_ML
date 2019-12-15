import avalanche.neuralnet.nets.ConvolutionalNeuralNet;
import avalanche.neuralnet.util.conv.dimred.ConvolutionLayer;
import avalanche.neuralnet.util.conv.dimred.PoolingLayer;
import avalanche.neuralnet.util.conv.dimred.ReLULayer;
import avalanche.neuralnet.util.conv.imgrep.MultiChannelImage;
import avalanche.util.ArrayUtils;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;

import static avalanche.util.ArrayUtils.a;

public class CNNTesting3 {

    public static void main(String[] args) throws IOException {

        // Make the neural net
        ConvolutionalNeuralNet cnn = new ConvolutionalNeuralNet();

        // Dimension Reduction layers
        cnn.addDimRedLayer(new ConvolutionLayer(1, 2, a(3, 3), 3)); // 1 * 2 = 2 outputs
        cnn.addDimRedLayer(new ReLULayer());
        cnn.addDimRedLayer(new ConvolutionLayer(2, 2, a(3, 3), 3)); // 2 * 2 = 4 outputs
        cnn.addDimRedLayer(new ReLULayer());
        cnn.addDimRedLayer(new ConvolutionLayer(4, 2, a(3, 3), 3)); // 4 * 2 = 8 outputs
        cnn.addDimRedLayer(new ReLULayer());
        cnn.addDimRedLayer(new PoolingLayer(3, 3));
        cnn.addDimRedLayer(new PoolingLayer(3, 3));
        cnn.addDimRedLayer(new PoolingLayer(3, 3));
        cnn.addDimRedLayer(new PoolingLayer(3, 3));

        // Fully connected layers
        cnn.addNeuronLayer(100, 1);
        cnn.addNeuronLayer(50, 100);
        cnn.addNeuronLayer(20, 50);
        cnn.addNeuronLayer(3,20);

        // Load the images
        // [1,0,0] = O, [0,1,0] = TRI, [0,0,1] = RECT
        File[] o_s    = new File("D:\\eclipse-workspace\\Machine_Learning\\_images\\_training images\\o").listFiles();
        File[] tri_s  = new File("D:\\eclipse-workspace\\Machine_Learning\\_images\\_training images\\triangle").listFiles();
        File[] rect_s = new File("D:\\eclipse-workspace\\Machine_Learning\\_images\\_training images\\rectangle").listFiles();

        MultiChannelImage[] mciO = new MultiChannelImage[o_s.length];
        MultiChannelImage[] mciT = new MultiChannelImage[tri_s.length];
        MultiChannelImage[] mciR = new MultiChannelImage[rect_s.length];

        for (int i=0; i<mciO.length; i++) {
            mciO[i] = new MultiChannelImage(o_s[i], true);
        }
        for (int i=0; i< mciT.length; i++) {
            mciT[i] = new MultiChannelImage(tri_s[i], true);
        }
        for (int i=0; i< mciR.length; i++) {
            mciR[i] = new MultiChannelImage(rect_s[i], true);
        }

        List<MultiChannelImage> trainingImages = ArrayUtils.concatenateList(mciO, mciT, mciR);

        double[][] outputTrainingArray = new double[mciO.length+mciT.length+mciR.length][3];
        for (int i=0; i<mciO.length; i++) {
            outputTrainingArray[i] = a(1D, 0D, 0D);
        }
        for (int k=mciO.length; k<mciO.length+mciT.length; k++) {
            outputTrainingArray[k] = a(0D, 1D, 0D);
        }
        for (int j=mciO.length+mciT.length; j<outputTrainingArray.length; j++) {
            outputTrainingArray[j] = a(0D, 0D, 1D); // TF? FIXME: ONLY THIS IS REGISTERING!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                                                                // This is the only thing being trained because it's the last thing being trained
        }

        for (double[] tArray : outputTrainingArray) {
            System.out.println(Arrays.toString(tArray));
        }

        cnn.train(trainingImages, outputTrainingArray, 100, 10);

        // Test the cnn
        MultiChannelImage o = new MultiChannelImage(new File("D:\\eclipse-workspace\\Machine_Learning\\_images\\_testing images\\o.png"), true);    // 1 0 0
        MultiChannelImage r = new MultiChannelImage(new File("D:\\eclipse-workspace\\Machine_Learning\\_images\\_training images\\o\\7.png"), true);    // 0 0 1

        System.out.println(Arrays.toString(cnn.classify(o)));
        System.out.println(Arrays.toString(cnn.classify(r)));

        // Dump the net
        ConvolutionalNeuralNet.saveNet(cnn, "ConvolutionalNeuralNet3.1.ava");
    }
}
