package avalanche.neuralnet.util.conv.dimred;

import avalanche.neuralnet.util.conv.imgrep.MultiChannelImage;
import avalanche.num.Matrix;

import java.util.ArrayList;
import java.util.List;

import static avalanche.util.ArrayUtils.a;

public class PoolingLayer implements DimRedLayer {

    private int stride;
    private int poolWidth;      // Width  = x
    private int poolHeight;     // Height = y
    private PoolingMethod poolingMethod;
    private boolean fork;                       // If fork, then we keep the number of channels inputted- otherwise, merge into 1

    public PoolingLayer(int poolWidth, int poolHeight) {
        this(poolWidth, poolHeight, 2, PoolingMethod.MAX_POOLING);
    }
    public PoolingLayer(int poolWidth, int poolHeight, PoolingMethod poolingMethod) {
        this(poolWidth, poolHeight, 2, poolingMethod);
    }
    public PoolingLayer(int poolWidth, int poolHeight, int stride) {
        this(poolWidth, poolHeight, stride, PoolingMethod.MAX_POOLING);
    }
    public PoolingLayer(int poolWidth, int poolHeight, int stride, PoolingMethod poolingMethod) {
        this.stride         = stride;
        this.poolWidth      = poolWidth;
        this.poolHeight     = poolHeight;
        this.poolingMethod  = poolingMethod;

        this.fork = true;                       // If the user wishes to change forking, use setFork()
    }

    public PoolingMethod getPoolingMethod() {
        return poolingMethod;
    }
    public int getStride() {
        return stride;
    }
    public boolean getFork() {
        return fork;
    }
    public void setPoolingMethod(PoolingMethod poolingMethod) {
        this.poolingMethod = poolingMethod;
    }
    public void setStride(int stride) {
        this.stride = stride;
    }
    public void setFork(boolean fork) {
        this.fork = fork;
    }

    public MultiChannelImage feedSingular(MultiChannelImage input) {
        return pool(input);
    }
    public List<MultiChannelImage> feedForward(List<MultiChannelImage> inputs) {

        List<MultiChannelImage> convolutionResults = new ArrayList<>();

        // Pool each input
        for (MultiChannelImage image : inputs) {
            convolutionResults.add(pool(image));
        }

        return convolutionResults;
    }

    private MultiChannelImage pool(MultiChannelImage image) {

        if (!fork) {

            List<List<Double>> poolImagePixelList = new ArrayList<>();
            double pixelOutput;     // The value that the current pixel being pooled will have, reset each iteration

            for (int r=0; r <= image.getHeight() - poolHeight; r += stride) {
                poolImagePixelList.add(new ArrayList<>());

                for (int c=0; c <= image.getWidth() - poolWidth; c += stride) {

                    pixelOutput = 0;    // We have reached a new pixel, therefore the output is different
                    for (int channelIndex = 0; channelIndex < image.getChannelCount(); channelIndex++) {

                        // Grab the chunk of the matrix that is being pooled
                        Matrix matrixBeingPooled = image.getChannel(channelIndex).slice(
                                a(r, r + poolHeight),
                                a(c, c + poolWidth)
                        );

                        // Depending on the pooling method, either find the max in the matrix or add all, or avg
                        switch (poolingMethod) {

                            case MAX_POOLING:
                                pixelOutput = matrixBeingPooled.findMax();
                                break;

                            case SUM_POOLING:
                                pixelOutput += matrixBeingPooled.sumUp();
                                break;

                            case MEAN_POOLING:
                                pixelOutput += matrixBeingPooled.findMean() / image.getChannelCount();  // Avg of all channels
                                break;
                        }
                    }

                    poolImagePixelList.get(poolImagePixelList.size() - 1).add(pixelOutput);
                }
            }

            // Convert the nested list to a nested array
            double[][] poolImagePixelArray = new double[poolImagePixelList.size()][poolImagePixelList.get(0).size()];
            for (int i = 0; i < poolImagePixelArray.length; i++) {
                for (int j = 0; j < poolImagePixelArray[0].length; j++) {
                    poolImagePixelArray[i][j] = poolImagePixelList.get(i).get(j);
                }
            }

            return new MultiChannelImage(Matrix.from2D(poolImagePixelArray));

        } else {

            List<List<List<Double>>> multiChannelPoolList = new ArrayList<>();

            // For each channel in the image, pool channel and add to mCPoolList
            for (Matrix channelMatrix : image.getChannels()) {
                multiChannelPoolList.add(new ArrayList<>());    // Add new channel

                for (int r=0; r <= image.getHeight() - poolHeight; r += stride) {
                    multiChannelPoolList.get(multiChannelPoolList.size()-1).add(new ArrayList<>());
                    for (int c=0; c<= image.getWidth() - poolWidth; c += stride) {

                        double pixelOutput = 0;

                        // Grab the section of the current channel that's getting pooled
                        Matrix matrixBeingPooled = channelMatrix.slice(
                                a(r, r + poolHeight),
                                a(c, c + poolWidth)
                        );

                        switch (poolingMethod) {

                            case MAX_POOLING:
                                pixelOutput = matrixBeingPooled.findMax();
                                break;

                            case SUM_POOLING:
                                pixelOutput = matrixBeingPooled.sumUp();
                                break;

                            case MEAN_POOLING:
                                pixelOutput = matrixBeingPooled.findMean();
                                break;
                        }

                        List<List<Double>> imagePixelList = multiChannelPoolList.get(multiChannelPoolList.size() - 1);
                        List<Double> imagePixelRow        = imagePixelList.get(imagePixelList.size()-1);
                        imagePixelRow.add(pixelOutput);
                    }
                }
            }

            // Convert multiChannelPoolList into a MultiChannelImage
            double[][][] multiChannelPoolArray =
                    new double[multiChannelPoolList.size()]
                            [multiChannelPoolList.get(0).size()]
                            [multiChannelPoolList.get(0).get(0).size()];

            for (int i=0; i < multiChannelPoolArray.length; i++) {
                for (int j=0; j < multiChannelPoolArray[0].length; j++) {
                    for (int k=0; k < multiChannelPoolArray[0][0].length; k++) {
                        multiChannelPoolArray[i][j][k] = multiChannelPoolList.get(i).get(j).get(k);
                    }
                }
            }

            Matrix[] multiChannelMatrix = new Matrix[multiChannelPoolArray.length];
            for (int i=0; i < multiChannelMatrix.length; i++) {
                multiChannelMatrix[i] = Matrix.from2D(multiChannelPoolArray[i]);
            }

            return new MultiChannelImage(multiChannelMatrix);

        }
    }
}
