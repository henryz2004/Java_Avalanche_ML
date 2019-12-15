package avalanche.neuralnet.nets;

import avalanche.neuralnet.util.Activation;
import avalanche.neuralnet.util.NeuronLayer;
import avalanche.neuralnet.util.conv.dimred.DimRedLayer;
import avalanche.neuralnet.util.conv.imgrep.MultiChannelImage;
import avalanche.num.Matrix;

import java.io.*;
import java.util.*;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.ThreadLocalRandom;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Collectors;

/**
 * Another tuple, this time Matrix - expected output in order to avoid concurrency problems
 */
class TrainingTuple {

    Matrix   associatedTrainingMatrix;
    double[] associatedTrainingOutput;

    public TrainingTuple(Matrix trainingMatrix, double[] trainingOutput) {
        associatedTrainingMatrix = trainingMatrix;
        associatedTrainingOutput = trainingOutput;
    }
}

/**
 * Simple comparable tuple (index - value)
 */
class Tuple<E> implements Comparable<Tuple<E>> {

    int index;
    E value;

    Tuple(int index, E value) {
        this.index = index;
        this.value = value;
    }

    @Override
    public int compareTo(Tuple<E> eTuple) {
        return index - eTuple.index;
    }
}

/**
 * Currently, the 'training' of a CNN is basically training the fully connected layers and not the filters themselves
 *
 * Because the fullyConnectedLayers must match up perfectly with the 'vectorized' output of the dimred layers, the
 * very first neuron layer's # of inputs will be automatically adjusted during training
 *
 * TODO: Potential normalization of inputs to fully connected ([0.2, 0.7, 0.6] -> [0, 1, 1])
 */
public class ConvolutionalNeuralNet implements Serializable {

    private List<DimRedLayer> dimensionReductionLayers;
    private NeuralNet         fullyConnectedLayers;

    public ConvolutionalNeuralNet() {
        dimensionReductionLayers = new ArrayList<>();
        fullyConnectedLayers     = new NeuralNet(Activation.SIGMOID);   // TODO: Not just sigmoid!
    }

    public void addDimRedLayer(DimRedLayer dimRedLayer) {
        dimensionReductionLayers.add(dimRedLayer);
    }
    public void addNeuronLayer(int neuronCount, int inputsPerNeuron) {
        fullyConnectedLayers.addLayer(neuronCount, inputsPerNeuron);
    }
    public void addNeuronLayer(NeuronLayer neuronLayer) {
        fullyConnectedLayers.addLayer(neuronLayer);
    }

    /**
     * Training the fully connected layers. Note that because right now ONLY training the FCL's are supported,
     * train() and trainFCL() are equivalent
     *
     * The chunk size is 1 image
     *
     * It is expected that the images are all of the same size; TODO: Add more flexibility, add batch size
     */
    public void train(List<MultiChannelImage> trainingImages, double[][] trainingOutputs, int iterations, int batchSize) {
        trainFCL(trainingImages, trainingOutputs, iterations, batchSize);
    }
    public void trainFCL(List<MultiChannelImage> trainingImages, double[][] trainingOutputs, int iterations, int batchSize) {

        long start = System.currentTimeMillis();
        System.out.println("Reducing image dimensions");

        // Pass images through dimred layers
        // Use threads for maximum speed
        // Convert trainingImages (List) -> concurrent queue of tuples
        ConcurrentLinkedQueue<Tuple<MultiChannelImage>> tupleTrainingQueue = new ConcurrentLinkedQueue<>();
        for (int i=0; i<trainingImages.size(); i++) {
            tupleTrainingQueue.add(new Tuple<>(i, trainingImages.get(i)));
        }
        List<Tuple<List<MultiChannelImage>>> convolutedTupledTrainingImages = Collections.synchronizedList(new ArrayList<>());      // This is where the reduced-dim images will be stored

        // Spawn the DimRedThreads
        List<DimRedThread> dimRedThreads = new ArrayList<>();
        for (int i=0; i<8; i++) {
            DimRedThread thread = new DimRedThread(tupleTrainingQueue, convolutedTupledTrainingImages);
            dimRedThreads.add(thread);
            thread.start();
        }
        for (DimRedThread thread : dimRedThreads) {
            try {
                thread.thread.join();
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }

        // Do I even need to sort the tuples?
        //Collections.sort(convolutedTupledTrainingImages);

        System.out.println("Reducing image dimensions [DONE] took " + (System.currentTimeMillis() - start) + " millis for " + trainingImages.size() + " images");

        List<Matrix> imageVectors  = new ArrayList<>();
        List<Matrix> outputVectors = new ArrayList<>();

        // Vectorize the training inputs and outputs
        for (Tuple<List<MultiChannelImage>> tupledTrainingImage : convolutedTupledTrainingImages) {
            imageVectors.addAll(vectorize(tupledTrainingImage.value));
            for (int j = 0; j < tupledTrainingImage.value.size(); j++) {
                outputVectors.add(Matrix.from1D(trainingOutputs[tupledTrainingImage.index]));
                System.out.println(Arrays.toString(trainingOutputs[tupledTrainingImage.index]));
            }
        }

        // Modify the first layer of the fullyConnectedLayers so the # of inputs matches up with the vectorized images
        NeuronLayer firstLayer = fullyConnectedLayers.removeLayer(0);
        firstLayer = new NeuronLayer(firstLayer.numNeurons, imageVectors.get(0).numCols());
        fullyConnectedLayers.addLayer(firstLayer, 0);   // Add the modified layer back into the stack

        System.out.println(Arrays.toString(firstLayer.synapticWeights.shape()));

        // Partition into batches for training
        // https://codereview.stackexchange.com/questions/27928/split-java-arraylist-into-equal-parts
        AtomicInteger c = new AtomicInteger(0);
        List<List<Matrix>> imageVectorBatches = new ArrayList<>(imageVectors.stream()
                .collect(Collectors.groupingBy( it -> c.getAndIncrement()/batchSize))
                .values());

        c.set(0);
        List<List<Matrix>> outputVectorBatches = new ArrayList<>(outputVectors.stream()
                .collect(Collectors.groupingBy( it -> c.getAndIncrement()/batchSize))
                .values());

        List<Matrix> imageMatrixBatches  = new ArrayList<>();
        List<Matrix> outputMatrixBatches = new ArrayList<>();

        for (int i=0; i<imageVectorBatches.size(); i++) {
            imageMatrixBatches.add(Matrix.fromRVectors(imageVectorBatches.get(i)));
            outputMatrixBatches.add(Matrix.fromRVectors(outputVectorBatches.get(i)));
        }

        // Train each batch
        for (int batchIndex = 0; batchIndex < imageMatrixBatches.size(); batchIndex++) {
            fullyConnectedLayers.train(imageMatrixBatches.get(batchIndex), outputMatrixBatches.get(batchIndex), iterations);
        }
    }

    public Matrix[] classify(MultiChannelImage inputImage) {

        // Feed through dim red layers and then vectorize
        List<MultiChannelImage> convolutedImage = feedDimRed(Collections.singletonList(inputImage)).get(0);
        List<Matrix> vectorizedImage = vectorize(convolutedImage);

        // Collect all classifications in array of matrices
        Matrix[] classifications = new Matrix[vectorizedImage.size()];

        try {
            for (int i = 0; i < vectorizedImage.size(); i++) {
                Matrix classification = fullyConnectedLayers.think(vectorizedImage.get(i));
                classifications[i] = classification;
            }

            return classifications;

        } catch (Exception e) {
            e.printStackTrace();
        }

        return null;
    }

    private List<Matrix> vectorize(List<MultiChannelImage> imageList) {

        List<Matrix> vectors = new ArrayList<>();

        for (MultiChannelImage image : imageList) {
            // [[R,G,B]]
            double[][] rgbFlat = new double[3][];
            rgbFlat[0] = image.getChannel(0).flatten();
            rgbFlat[1] = image.getChannel(1).flatten();
            rgbFlat[2] = image.getChannel(2).flatten();

            double[] rgbFlatVector = new double[3*rgbFlat[0].length];
            int k=0;
            for (int i=0; i<3; i++) {
                for (int j=0; j<rgbFlat[i].length; j++) {
                    rgbFlatVector[k] = rgbFlat[i][j];
                    k++;
                }
            }

            // Convert rgbFlatVector to Matrix and add to vectors
            vectors.add(Matrix.from1D(rgbFlatVector));
        }
        return vectors;
    }
    private List<MultiChannelImage> feedDimRed(MultiChannelImage image) {

        List<MultiChannelImage> convOut = Collections.singletonList(image);
        for (DimRedLayer dimRedLayer : dimensionReductionLayers) {
            convOut = dimRedLayer.feedForward(convOut);
        }
        return convOut;
    }
    private List<List<MultiChannelImage>> feedDimRed(List<MultiChannelImage> images) {

        // List of List -> first (outer) list organizes outputs by original image, and the inner contains
        List<List<MultiChannelImage>> convolutedImages = new ArrayList<>();

        // Pass images through dimred layers
        for (MultiChannelImage image1 : images) {
            List<MultiChannelImage> image = Collections.singletonList(image1);
            for (DimRedLayer dimRedLayer : dimensionReductionLayers) {
                image = dimRedLayer.feedForward(image);
            }
            convolutedImages.add(image);
        }

        return convolutedImages;
    }

    // Static method used to dump nets and load them
    public static void saveNet(ConvolutionalNeuralNet net, String fileName) {

        try {
            FileOutputStream f = new FileOutputStream(fileName);
            BufferedOutputStream b = new BufferedOutputStream(f);
            ObjectOutputStream o = new ObjectOutputStream(b);

            o.writeObject(net);

            o.close();
            b.close();
            f.close();

        } catch (IOException e) {
            e.printStackTrace();
        }

    }
    public static ConvolutionalNeuralNet loadNet(String fileName) {

        try {
            FileInputStream     f = new FileInputStream(fileName);
            BufferedInputStream b = new BufferedInputStream(f);
            ObjectInputStream   o = new ObjectInputStream(b);

            ConvolutionalNeuralNet net = (ConvolutionalNeuralNet) o.readObject();

            o.close();
            b.close();
            f.close();

            return net;

        } catch (IOException | ClassNotFoundException e) {
            e.printStackTrace();
        }

        return null;
    }

    public class DimRedThread implements Runnable {

        Thread thread;
        ConcurrentLinkedQueue<Tuple<MultiChannelImage>> imageQueue;
        List<Tuple<List<MultiChannelImage>>> reducedImages; // Ignore the tuple part

        DimRedThread(ConcurrentLinkedQueue<Tuple<MultiChannelImage>> imageQueue, List<Tuple<List<MultiChannelImage>>> reducedImages) {
            this.thread        = new Thread(this, "dimred thread");
            this.imageQueue    = imageQueue;
            this.reducedImages = reducedImages;
        }

        public void run() {

            while (!imageQueue.isEmpty()) {
                Tuple<MultiChannelImage> imageTuple = imageQueue.remove(); // Take from the queue

                // Feed through dimred layers and add to reducedImages
                reducedImages.add(new Tuple<>(imageTuple.index, feedDimRed(imageTuple.value)));
            }
        }

        public void start() {
            thread.start();
        }
    }
}
