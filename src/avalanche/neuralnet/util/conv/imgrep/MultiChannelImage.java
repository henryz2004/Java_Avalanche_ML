package avalanche.neuralnet.util.conv.imgrep;

import avalanche.num.Matrix;
import avalanche.num.util.MathUtils;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Simply just a container class for List<Matrix>
 * A matrix is a 'channel'
 * The values in the matrix are the values for the image
 *
 * Each channel can represent a different color channel, eg. R, G, B
 */
public class MultiChannelImage implements Serializable {

    private List<Matrix> image;     // Stacked from front to back

    public MultiChannelImage() {
        image = new ArrayList<>();
    }
    public MultiChannelImage(Matrix singleChannelMatrix) {
        image = new ArrayList<>();
        image.add(singleChannelMatrix);
    }
    public MultiChannelImage(Matrix[] multipleChannelMatrix) {
        image = new ArrayList<>(Arrays.asList(multipleChannelMatrix));
    }
    public MultiChannelImage(File imageFile, boolean normalizeRGB) throws IOException {
        // RGB no alpha
        image = new ArrayList<>(3);     // 3 channels

        // Load the image associated with the imageFile
        BufferedImage bufferedImage = ImageIO.read(imageFile);

        int width  = bufferedImage.getWidth();
        int height = bufferedImage.getHeight();

        Matrix R = Matrix.fillEmpty(height, width, 0);      // Rows are stacked vertically, therefore rows = height, cols = width
        Matrix G = Matrix.fillEmpty(height, width, 0);
        Matrix B = Matrix.fillEmpty(height, width, 0);

         // For each coordinate in the image, find the R, G, B values, and set that value inside RGB
        for (int row=0; row<height; row++) {
            for (int col=0; col<width; col++) {

                Color color = new Color(bufferedImage.getRGB(col, row));

                // Fill in the channel matrices
                double r = color.getRed();
                double g = color.getGreen();
                double b = color.getBlue();

                R.setAt(row, col, normalizeRGB ? r/255 : r);
                G.setAt(row, col, normalizeRGB ? g/255 : g);
                B.setAt(row, col, normalizeRGB ? b/255 : b);
            }
        }

        image.add(R);
        image.add(G);
        image.add(B);
    }

    public void addChannel(Matrix newChannelMatrix) {
        image.add(newChannelMatrix);
    }
    public Matrix removeChannel(int index) {
        return image.remove(index);
    }
    public Matrix popChannel() {
        return image.remove(image.size()-1);
    }

    public List<Matrix> getChannels() {
        return image;
    }
    public Matrix getChannel(int index) {
        return image.get(index);
    }
    public int getChannelCount() {
        return image.size();
    }
    public int getHeight() {
        return image.size() > 0 ? image.get(0).numRows() : -1;
    }
    public int getWidth() {
        return image.size() > 0 ? image.get(0).numCols() : -1;
    }
    public static void printImage(MultiChannelImage mci) {
        List<Matrix> channels = mci.getChannels();

        System.out.println("Printing image with " + channels.size() + " channels");

        for (int channel=0; channel<channels.size(); channel++) {
            Matrix matrix = channels.get(channel);

            System.out.println("Channel " + channel);

            double[][] nestedArray = matrix.toArray();
            for (double[] doubles : nestedArray) {
                double[] rounded = new double[doubles.length];
                for (int i=0; i<rounded.length; i++) {
                    rounded[i] = MathUtils.round(doubles[i], 2);
                }
                System.out.println(Arrays.toString(rounded));
            }
            System.out.println();
        }
    }

    @Override
    public String toString() {
        String channels = image.size() + " channels, ";
        String suffix = "";
        if (image.size()>0) suffix = image.get(0).numCols() + "x" + image.get(0).numRows();
        return channels + suffix;
    }
}
