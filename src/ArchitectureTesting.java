import avalanche.neuralnet.nets.NeuralNet;
import avalanche.neuralnet.util.architecture.LayerArchitecture;
import avalanche.neuralnet.util.architecture.NetArchitecture;

public class ArchitectureTesting {

    public static void main(String[] args) throws Exception{

        NetArchitecture architecture = new NetArchitecture();
        architecture.addLayer(new LayerArchitecture(4, 3));
        architecture.addLayer(new LayerArchitecture(1, 4));

        NeuralNet net  = architecture.constructNet();
        NeuralNet net2 = architecture.constructNet();

        try {
            Thread.sleep(1000);                 //1000 milliseconds is one second.
        } catch(InterruptedException ex) {
            ex.printStackTrace();
            Thread.currentThread().interrupt();
        }

        NeuralNet net3 = architecture.constructNet();

        net.printLayers(true);
        net2.printLayers(true);
        net3.printLayers(true);
    }
}
