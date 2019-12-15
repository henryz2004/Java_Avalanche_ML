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

public class CNNTesting {

    public static void main(String[] args) throws IOException {

        // Make the neural net
        ConvolutionalNeuralNet cnn = new ConvolutionalNeuralNet();

        // Dimension Reduction layers
        cnn.addDimRedLayer(new ConvolutionLayer(1, 2, a(3, 3), 3)); // 1 * 2 = 2 outputs
        cnn.addDimRedLayer(new ReLULayer());
        cnn.addDimRedLayer(new ConvolutionLayer(2, 1, a(3, 3), 3)); // 2 * 1 = 2 outputs
        cnn.addDimRedLayer(new ReLULayer());
        cnn.addDimRedLayer(new ConvolutionLayer(2, 2, a(2, 2), 3)); // 2 * 2 = 4 outputs
        cnn.addDimRedLayer(new ReLULayer());
        cnn.addDimRedLayer(new PoolingLayer(3, 3));
        cnn.addDimRedLayer(new PoolingLayer(3, 3));
        cnn.addDimRedLayer(new PoolingLayer(3, 3));
        cnn.addDimRedLayer(new PoolingLayer(3, 3));

        // Fully connected layers
        cnn.addNeuronLayer(90, 1);
        cnn.addNeuronLayer(50, 90);
        cnn.addNeuronLayer(40, 50);
        cnn.addNeuronLayer(30, 40);
        cnn.addNeuronLayer(2, 30);   // First on = x, second on = o

        // Load the images
        File[] x_s = new File("D:\\eclipse-workspace\\Machine_Learning\\_images\\_training images\\x").listFiles();
        File[] o_s = new File("D:\\eclipse-workspace\\Machine_Learning\\_images\\_training images\\o").listFiles();

        System.out.println(x_s.length + " " + o_s.length);

        MultiChannelImage[] mciX = new MultiChannelImage[x_s.length];
        MultiChannelImage[] mciO = new MultiChannelImage[o_s.length];

        for (int i=0; i<mciX.length; i++) {
            mciX[i] = new MultiChannelImage(x_s[i], true);
            mciO[i] = new MultiChannelImage(o_s[i], true);
        }

        List<MultiChannelImage> trainingImages = ArrayUtils.concatenateList(mciX, mciO);

        // Make output training matrix
        // X = [1, 0], O = [0, 1]
        double[][] outputTrainingArray = new double[mciX.length+mciO.length][2];
        for (int i=0; i<mciX.length; i++) {
            outputTrainingArray[i] = a(1D, 0D);
        }
        for (int j=mciX.length; j<outputTrainingArray.length; j++) {
            outputTrainingArray[j] = a(0D, 1D);
        }

        cnn.train(trainingImages, outputTrainingArray, 10, 1100);

        // Test the cnn
        MultiChannelImage x = new MultiChannelImage(new File("D:\\eclipse-workspace\\Machine_Learning\\_images\\_testing images\\x.png"), true);
        MultiChannelImage o = new MultiChannelImage(new File("D:\\eclipse-workspace\\Machine_Learning\\_images\\_testing images\\o.png"), true);

        System.out.println(Arrays.toString(cnn.classify(x)));
        System.out.println(Arrays.toString(cnn.classify(trainingImages.get(0))));   //x
        System.out.println(Arrays.toString(cnn.classify(o)));

        // Dump the net
        ConvolutionalNeuralNet.saveNet(cnn, "ConvolutionalNeuralNet2.ava");
    }
}
