package avalanche.neuralnet.util.conv.dimred;

import avalanche.neuralnet.util.conv.imgrep.MultiChannelImage;

import java.io.Serializable;
import java.util.List;

/**
 * DimRedLayer stands for "Dimension Reduction Layer", but Dim-red also sounds pretty nice
 */
public interface DimRedLayer extends Serializable {

    public List<MultiChannelImage> feedForward(List<MultiChannelImage> inputs);
}
