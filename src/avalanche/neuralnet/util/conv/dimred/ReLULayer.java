package avalanche.neuralnet.util.conv.dimred;

import avalanche.neuralnet.util.activation.ReLU;
import avalanche.neuralnet.util.conv.imgrep.MultiChannelImage;
import avalanche.num.Expression;
import avalanche.num.Matrix;

import java.util.ArrayList;
import java.util.List;

/**
 * Although Rectified Linear Units do not actually reduce dim. (DimRed), it is still implemented
 * for ease of implementation
 */
public class ReLULayer implements DimRedLayer {

    private Expression reluExpression;

    public ReLULayer() {
        reluExpression = new ReLU();
    }

    /**
     * Apply ReLU to each channel in each image
     * @param inputs
     * @return
     */
    public List<MultiChannelImage> feedForward(List<MultiChannelImage> inputs) {

        List<MultiChannelImage> reluResults = new ArrayList<>();

        for (MultiChannelImage mci : inputs) {
            MultiChannelImage reluImage = new MultiChannelImage();
            for (Matrix channel    : mci.getChannels()) {
                Matrix reluMatrix = channel.useExpression(reluExpression);
                reluImage.addChannel(reluMatrix);
            }
            reluResults.add(reluImage);
        }

        return reluResults;
    }
}
