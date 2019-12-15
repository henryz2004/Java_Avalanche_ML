package avalanche.neuralnet.util.architecture;

import avalanche.neuralnet.nets.NeuralNet;
import avalanche.neuralnet.util.Activation;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

public class NetArchitecture {

    private long seed;

    public List<LayerArchitecture> architecture;

    // Constructors. Seed can be used to generate identical nets
    public NetArchitecture() {
         architecture = new ArrayList<>();
         seed = -1;
    }
    public NetArchitecture(long netSeed){
        this();
        seed = netSeed;
    }
    public NetArchitecture(LayerArchitecture[] layerArchitectures) {
        architecture = Arrays.stream(layerArchitectures)
                .collect(Collectors.toList());
    }
    public NetArchitecture(LayerArchitecture[] layerArchitectures, long netSeed) {
        this(layerArchitectures);
        seed = netSeed;
    }
    public NetArchitecture(List<LayerArchitecture> layerArchitectures) {
        architecture = layerArchitectures;
    }
    public NetArchitecture(List<LayerArchitecture> layerArchitectures, long netSeed) {
        this(layerArchitectures);
        seed = netSeed;
    }

    public void addLayer(LayerArchitecture layerArchitecture) {
        architecture.add(layerArchitecture);
    }

    public NeuralNet constructNet() {

        NeuralNet net = new NeuralNet(Activation.SIGMOID);  // Setting the learning rate is useless; we're not bping

        for (LayerArchitecture layerArchitecture : architecture) {
            net.addLayer(layerArchitecture.constructLayer(seed));
        }

        return net;
    }
}
