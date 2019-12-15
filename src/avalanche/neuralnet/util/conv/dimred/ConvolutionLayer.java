package avalanche.neuralnet.util.conv.dimred;

import avalanche.neuralnet.util.conv.imgrep.MultiChannelImage;
import avalanche.num.Matrix;

import java.util.ArrayList;
import java.util.List;

import static avalanche.util.ArrayUtils.a;

public class ConvolutionLayer implements DimRedLayer {

    private int stride;
    private MultiChannelImage[][] filters;      // List of lists of filters to apply to each input
    private boolean fork;                       // If fork, then we keep the number of channels inputted- otherwise, merge into 1

    // TODO:  Specialized constructors for predefined edge and corner detection, etc.
    // Stride isn't offered as a parameter for the constructor because it's not as important for convolution
    public ConvolutionLayer(int numberOfInputs, int filtersPerInput, int[] filterDimensions, int channels) {
        stride  = 1;
        filters = new MultiChannelImage[numberOfInputs][filtersPerInput];
        fork    = true;

        // Initialize the filters randomly
        for (int i=0; i<numberOfInputs; i++) {
            for (int j=0; j<filtersPerInput; j++) {

                MultiChannelImage filter = new MultiChannelImage();

                for (int k=0; k<channels; k++) {
                    filter.addChannel(Matrix.fillRandom(filterDimensions[0], filterDimensions[1], -5, 5));
                }
                filters[i][j] = filter;
            }
        }
    }
    public ConvolutionLayer(MultiChannelImage[][] userSpecifiedFilters) {
        stride  = 1;
        filters = userSpecifiedFilters;
        fork    = true;
    }

    public int getStride() {
        return stride;
    }
    public void setStride(int stride) {
        this.stride = stride;
    }

    public MultiChannelImage[][] getFilters() {
        return filters;
    }
    public boolean getFork() {
        return fork;
    }
    public void setFilters(MultiChannelImage[][] filters) {
        this.filters = filters;
    }
    public void setFork(boolean fork) {
        this.fork = fork;
    }

    /**
     * Apply corresponding arrays of filters to each MultiChannelImage
     */
    public List<MultiChannelImage> feedForward(List<MultiChannelImage> inputs) {

        List<MultiChannelImage> convolutionResults = new ArrayList<>();

        for (int i=0; i<inputs.size(); i++) {
            MultiChannelImage image = inputs.get(i);

            // Apply all the filters at filters[i] to inputs. Fork results
            MultiChannelImage[] applicableFilters = filters[i];

            // It is expected that the number of channels in the inputs is the equivalent to the number of channels the
            // applicable filters have
            for (MultiChannelImage filter : applicableFilters) {
                convolutionResults.add(applyFilter(image, filter));
            }
        }

        return convolutionResults;
    }

    /**
     *
     * @param image     - The image being convoluted
     * @param filter    - The filter being used to convolve
     * @return the convoluted image (if fork then same # of channels as {@param image} otherwise 1 channel)
     */
    private MultiChannelImage applyFilter(MultiChannelImage image, MultiChannelImage filter) {

        if (!fork) {

            List<List<Double>> convImagePixelList = new ArrayList<>();  // List of lists of pixels (lists of rows, to be turned into a matrix)

            double pixelOutput;     // The value that the current pixel being convoluted will have, reset each iteration

            // (R, C) is where the topleft pixel of the filter will be placed
            for (int r = 0; r <= image.getHeight() - filter.getHeight(); r += stride) {
                convImagePixelList.add(new ArrayList<>());

                for (int c = 0; c <= image.getWidth() - filter.getWidth(); c += stride) {

                    pixelOutput = 0;    // We have reached a new pixel, therefore the output is different
                    for (int channelIndex = 0; channelIndex < image.getChannelCount(); channelIndex++) {

                        // Grab the filter and the chunk of the matrix the filter is filtering
                        Matrix channelFilterMatrix = filter.getChannel(channelIndex);
                        Matrix matrixBeingFiltered = image.getChannel(channelIndex).slice(
                                a(r, r + filter.getHeight()),
                                a(c, c + filter.getWidth())
                        );

                        // The raw product of multiplying the two matrices
                        Matrix filterProduct = Matrix.multiply(channelFilterMatrix, matrixBeingFiltered);

                        pixelOutput += filterProduct.findMean();   // Combine the results of all the channels; compress into 1 channel
                    }

                    convImagePixelList.get(convImagePixelList.size() - 1).add(pixelOutput);
                }
            }

            // Convert the nested list to a nested array
            double[][] convImagePixelArray = new double[convImagePixelList.size()][convImagePixelList.get(0).size()];
            for (int i = 0; i < convImagePixelArray.length; i++) {
                for (int j = 0; j < convImagePixelArray[0].length; j++) {
                    convImagePixelArray[i][j] = convImagePixelList.get(i).get(j);
                }
            }

            MultiChannelImage filteredImage = new MultiChannelImage(Matrix.from2D(convImagePixelArray));

            return filteredImage;

        } else {

            List<List<List<Double>>> multiChannelConvList = new ArrayList<>();

            // Make a new channel for each channel and add to the multiChannelConvList
            for (int channelIndex = 0; channelIndex < image.getChannelCount(); channelIndex++) {
                multiChannelConvList.add(new ArrayList<>());        // New channel

                for (int r=0; r <= image.getHeight() - filter.getHeight(); r += stride) {
                    multiChannelConvList.get(multiChannelConvList.size()-1).add(new ArrayList<>());
                    for (int c=0; c <= image.getWidth() - filter.getWidth(); c += stride) {

                        // Grab the section of the current channel that's getting filtered
                        Matrix matrixBeingFiltered = image.getChannel(channelIndex).slice(
                                a(r, r + filter.getHeight()),
                                a(c, c + filter.getWidth())
                        );

                        Matrix filterProduct = Matrix.multiply(filter.getChannel(channelIndex), matrixBeingFiltered);

                        double convResult = filterProduct.findMean();

                        // Add convResult to multiChannelConvList
                        List<List<Double>> imagePixelList = multiChannelConvList.get(multiChannelConvList.size() - 1);
                        List<Double> imagePixelRow        = imagePixelList.get(imagePixelList.size()-1);
                        imagePixelRow.add(convResult);
                    }
                }
            }

            // The conversion
            double[][][] multiChannelConvArray =
                    new double[multiChannelConvList.size()]
                            [multiChannelConvList.get(0).size()]
                            [multiChannelConvList.get(0).get(0).size()];

            for (int i=0; i < multiChannelConvArray.length; i++) {
                for (int j=0; j < multiChannelConvArray[0].length; j++) {
                    for (int k=0; k < multiChannelConvArray[0][0].length; k++) {
                        multiChannelConvArray[i][j][k] = multiChannelConvList.get(i).get(j).get(k);
                    }
                }
            }

            Matrix[] multiChannelMatrix = new Matrix[multiChannelConvArray.length];
            for (int i=0; i < multiChannelMatrix.length; i++) {
                multiChannelMatrix[i] = Matrix.from2D(multiChannelConvArray[i]);
            }

            return new MultiChannelImage(multiChannelMatrix);
        }
    }
}
