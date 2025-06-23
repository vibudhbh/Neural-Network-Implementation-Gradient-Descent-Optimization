package network;

import java.util.List;

import data.Image;
import data.ImageDataSet;

import util.Log;


public class ConvolutionalNeuralNetwork {
    //this is the loss function for the output of the neural network
    LossFunction lossFunction;

    //this is the total number of weights in the neural network
    int numberWeights;

    //specifies if the CNN will use dropout
    boolean useDropout;
    //the dropout for nodes in the input layer
    double inputDropoutRate;
    //the dropout for nodes in the hidden layer
    double hiddenDropoutRate;

    //specify if the CNN will use batch normalization
    boolean useBatchNormalization;

    //the alpha value used to calculate the running
    //averages for batch normalization
    double alpha;

    //layers contains all the nodes in the neural network
    ConvolutionalNode[][] layers;

    public void createSmallNoPool(ActivationType activationType, int batchSize, int inputChannels, int inputY, int inputX, int inputPadding, int outputLayerSize) throws NeuralNetworkException {
        layers = new ConvolutionalNode[5][];

        layers[0] = new ConvolutionalNode[1];
        ConvolutionalNode inputNode = new ConvolutionalNode(0, 0, NodeType.INPUT, activationType /*doesn't matter for input*/, inputPadding /*padding*/ , batchSize, inputChannels, inputY, inputX, useDropout, inputDropoutRate, false, 0.0);
        layers[0][0] = inputNode;


        //first hidden layer has 4 nodes
        layers[1] = new ConvolutionalNode[4];
        for (int i = 0; i < 4; i++) {
            ConvolutionalNode node = new ConvolutionalNode(1, i, NodeType.HIDDEN, activationType, 0 /*padding*/ , batchSize, 1, 20, 20, useDropout, hiddenDropoutRate, useBatchNormalization, alpha);
            layers[1][i] = node;

            numberWeights += node.getNumberWeights();

            new ConvolutionalEdge(layers[0][0], node, inputChannels /* will be 1 for MNIST or 3 for CIFAR-10 to get the other feature maps to only have 1 channel*/, 13, 13);

            numberWeights += inputChannels * 13 * 13;
        }


        //second hidden layer also has 4 nodes, because it's just the max pooling from the first
        layers[2] = new ConvolutionalNode[4];
        for (int i = 0; i < 4; i++) {
            ConvolutionalNode node = new ConvolutionalNode(2, i, NodeType.HIDDEN, activationType, 0 /*padding*/ , batchSize, 1, 10, 10, useDropout, hiddenDropoutRate, useBatchNormalization, alpha);
            layers[2][i] = node;

            numberWeights += node.getNumberWeights();

            for (int j = 0; j < 4; j++) {
                new ConvolutionalEdge(layers[1][j], node, 1, 11, 11); //11x11 to get down to 10x10

                numberWeights += 11 * 11;
            }
        }


        //third hidden layer is dense with 10 nodes
        layers[3] = new ConvolutionalNode[10];
        for (int i = 0; i < 10; i++) {
            ConvolutionalNode node = new ConvolutionalNode(3, i, NodeType.HIDDEN, activationType, 0 /*padding*/ , batchSize, 1, 1, 1, useDropout, hiddenDropoutRate, useBatchNormalization, alpha);
            layers[3][i] = node;

            numberWeights += node.getNumberWeights();

            for (int j = 0; j < 4; j++) {
                new ConvolutionalEdge(layers[2][j], node, 1, 10, 10);

                numberWeights += 10 * 10;
            }
        }

        //output layer is dense with 10 nodes
        layers[4] = new ConvolutionalNode[10];
        for (int i = 0; i < 10; i++) {
            //no dropout or BN on output nodes
            ConvolutionalNode node = new ConvolutionalNode(4, i, NodeType.OUTPUT, activationType, 0 /*padding*/ , batchSize, 1, 1, 1, false, 0.0, false, 0.0); 
            layers[4][i] = node;

            //no bias on output nodes

            for (int j = 0; j < 10; j++) {
                new ConvolutionalEdge(layers[3][j], node, 1, 1, 1);

                numberWeights += 1;
            }
        }
    }

    public void createSmall(ActivationType activationType, int batchSize, int inputChannels, int inputY, int inputX, int inputPadding, int outputLayerSize) throws NeuralNetworkException {
        layers = new ConvolutionalNode[5][];

        layers[0] = new ConvolutionalNode[1];
        ConvolutionalNode inputNode = new ConvolutionalNode(0, 0, NodeType.INPUT, activationType /*doesn't matter for input*/, inputPadding /*padding*/ , batchSize, inputChannels, inputY, inputX, useDropout, inputDropoutRate, false, 0.0);
        layers[0][0] = inputNode;


        //first hidden layer has 4 nodes
        layers[1] = new ConvolutionalNode[4];
        for (int i = 0; i < 4; i++) {
            ConvolutionalNode node = new ConvolutionalNode(1, i, NodeType.HIDDEN, activationType, 0 /*padding*/ , batchSize, 1, 20, 20, useDropout, hiddenDropoutRate, useBatchNormalization, alpha);
            layers[1][i] = node;

            numberWeights += node.getNumberWeights();

            new ConvolutionalEdge(layers[0][0], node, inputChannels /* will be 1 for MNIST or 3 for CIFAR-10 to get the other feature maps to only have 1 channel*/, 13, 13);

            numberWeights += inputChannels * 13 * 13;
        }


        //second hidden layer also has 4 nodes, because it's just the max pooling from the first
        layers[2] = new ConvolutionalNode[4];
        for (int i = 0; i < 4; i++) {
            ConvolutionalNode node = new ConvolutionalNode(2, i, NodeType.HIDDEN, activationType, 0 /*padding*/ , batchSize, 1, 10, 10, useDropout, hiddenDropoutRate, useBatchNormalization, alpha);
            layers[2][i] = node;

            numberWeights += node.getNumberWeights();

            new PoolingEdge(layers[1][i], node, 2, 2); //stride of 2 and pool size of 2 for this max pooling operation
        }


        //third hidden layer is dense with 10 nodes
        layers[3] = new ConvolutionalNode[10];
        for (int i = 0; i < 10; i++) {
            ConvolutionalNode node = new ConvolutionalNode(3, i, NodeType.HIDDEN, activationType, 0 /*padding*/ , batchSize, 1, 1, 1, useDropout, hiddenDropoutRate, useBatchNormalization, alpha);
            layers[3][i] = node;

            numberWeights += node.getNumberWeights();

            for (int j = 0; j < 4; j++) {
                new ConvolutionalEdge(layers[2][j], node, 1, 10, 10);

                numberWeights += 10 * 10;
            }
        }

        //output layer is dense with 10 nodes
        layers[4] = new ConvolutionalNode[10];
        for (int i = 0; i < 10; i++) {
            //no dropout or BN on output nodes
            ConvolutionalNode node = new ConvolutionalNode(4, i, NodeType.OUTPUT, activationType, 0 /*padding*/ , batchSize, 1, 1, 1, false, 0.0, false, 0.0);
            layers[4][i] = node;

            //no bias on output nodes

            for (int j = 0; j < 10; j++) {
                new ConvolutionalEdge(layers[3][j], node, 1, 1, 1);

                numberWeights += 1;
            }
        }
    }

    public void createLeNet5(ActivationType activationType, int batchSize, int inputChannels, int inputY, int inputX, int inputPadding, int outputLayerSize) throws NeuralNetworkException {

        //make sure dropout is turned off on the last hidden layer
        // We'll create 8 layers total: 0..7
        layers = new ConvolutionalNode[8][];

        // ─────────────────────────────────────────────────────────────────
        // Layer 0: INPUT
        // 28×28 input + 2 padding => effectively 32×32
        // ─────────────────────────────────────────────────────────────────
        layers[0] = new ConvolutionalNode[1];
        ConvolutionalNode inputNode = new ConvolutionalNode(
                0, 0,
                NodeType.INPUT,
                activationType,  // doesn't matter for input
                inputPadding,    // e.g. 2
                batchSize,
                inputChannels,   // e.g. 1
                inputY,          // 28
                inputX,          // 28
                useDropout,
                inputDropoutRate,
                /*useBatchNormalization=*/false,
                /*alpha=*/0.0
        );
        layers[0][0] = inputNode;

        // ─────────────────────────────────────────────────────────────────
        // Layer 1: CONV1 => 6 maps, each 28×28
        // 5×5 filters from 32×32 => (32-5+1=28)
        // ─────────────────────────────────────────────────────────────────
        layers[1] = new ConvolutionalNode[6];
        int conv1OutY = (inputY + 2*inputPadding) - 5 + 1; // 28
        int conv1OutX = (inputX + 2*inputPadding) - 5 + 1; // 28
        for (int i = 0; i < 6; i++) {
            // This node has sizeZ=1, sizeY=28, sizeX=28
            // It uses a per-cell bias array => 1×28×28 biases
            ConvolutionalNode conv1Node = new ConvolutionalNode(
                    1, i,
                    NodeType.HIDDEN,
                    activationType,
                    0,           // no additional padding here
                    batchSize,
                    1,           // sizeZ
                    conv1OutY,   // 28
                    conv1OutX,   // 28
                    useDropout,
                    hiddenDropoutRate,
                    useBatchNormalization,
                    alpha
            );
            layers[1][i] = conv1Node;
            numberWeights += conv1Node.getNumberWeights(); // per-cell biases => 28×28 = 784 biases

            // Each 5×5 filter => 25 weights
            new ConvolutionalEdge(inputNode, conv1Node, inputChannels, 5, 5);
            numberWeights += (inputChannels * 5 * 5); // e.g. 25 if inputChannels=1
        }

        // ─────────────────────────────────────────────────────────────────
        // Layer 2: POOL1 => 6 maps, each 14×14 (2×2 pooling)
        // from 28×28 => 14×14
        // ─────────────────────────────────────────────────────────────────
        layers[2] = new ConvolutionalNode[6];
        for (int i = 0; i < 6; i++) {
            // This node has sizeZ=1, sizeY=14, sizeX=14
            // => 1×14×14 biases
            ConvolutionalNode pool1Node = new ConvolutionalNode(
                    2, i,
                    NodeType.HIDDEN,
                    activationType,
                    0,        // no extra padding
                    batchSize,
                    1,        // sizeZ
                    conv1OutY / 2, // 14
                    conv1OutX / 2, // 14
                    useDropout,
                    hiddenDropoutRate,
                    useBatchNormalization,
                    alpha
            );
            layers[2][i] = pool1Node;
            numberWeights += pool1Node.getNumberWeights(); // 14×14 = 196 biases

            // 2×2 max pool => no weights
            new PoolingEdge(layers[1][i], pool1Node, 2, 2);
        }

        // ─────────────────────────────────────────────────────────────────
        // Layer 3: CONV2 => 16 maps, each 10×10
        // Standard partial connectivity => 53 connections total
        // 5×5 filters => (14-5+1=10)
        // ─────────────────────────────────────────────────────────────────
        layers[3] = new ConvolutionalNode[16];
        // The classic partial connectivity table from the original LeNet-5
        int[][] partialTable = {
                {0, 1, 2},    // Row 0: 3 connections
                {1, 2, 3},    // Row 1: 3
                {2, 3, 4},    // Row 2: 3
                {3, 4, 5},    // Row 3: 3
                {0, 4, 5},    // Row 4: 3
                {0, 1, 5},    // Row 5: 3
                {0, 1, 2, 3},    // Row 6: 4
                {1, 2, 3, 4},    // Row 7: 4
                {2, 3, 4, 5},    // Row 8: 4
                {0, 3, 4, 5},    // Row 9: 4
                {0,1,4, 5},    // Row 10: 4
                {0,1,2,5},    // Row 11: 4
                {0, 1, 3, 4},    // Row 12: 4
                {1, 2, 4, 5},    // Row 13: 4
                {0, 2, 3, 5},    // Row 14: 4
                {0, 1, 2, 3, 4,5}  // Row 15: 5 (or 6 if original uses all) // row 15: 5 connections
        };
        for (int i = 0; i < 16; i++) {
            // Each map => sizeZ=1, sizeY=10, sizeX=10 => 1×10×10 biases
            ConvolutionalNode conv2Node = new ConvolutionalNode(
                    3, i,
                    NodeType.HIDDEN,
                    activationType,
                    0,
                    batchSize,
                    1,
                    14 - 5 + 1, // 10
                    14 - 5 + 1, // 10
                    useDropout,
                    hiddenDropoutRate,
                    useBatchNormalization,
                    alpha
            );
            layers[3][i] = conv2Node;
            numberWeights += conv2Node.getNumberWeights(); // 10×10=100 biases

            // Connect from the partial set of input maps
            for (int inMap : partialTable[i]) {
                new ConvolutionalEdge(layers[2][inMap], conv2Node, 1, 5, 5);
                numberWeights += 25; // 5×5=25 weights per connection
            }
        }

        // ─────────────────────────────────────────────────────────────────
        // Layer 4: POOL2 => 16 maps, each 5×5
        // 2×2 max pool => 10×10 => 5×5
        // ─────────────────────────────────────────────────────────────────
        layers[4] = new ConvolutionalNode[16];
        for (int i = 0; i < 16; i++) {
            // sizeZ=1, sizeY=5, sizeX=5 => 25 biases
            ConvolutionalNode pool2Node = new ConvolutionalNode(
                    4, i,
                    NodeType.HIDDEN,
                    activationType,
                    0,
                    batchSize,
                    1,
                    10 / 2, // 5
                    10 / 2, // 5
                    useDropout,
                    hiddenDropoutRate,
                    useBatchNormalization,
                    alpha
            );
            layers[4][i] = pool2Node;
            numberWeights += pool2Node.getNumberWeights(); // 5×5=25 biases

            new PoolingEdge(layers[3][i], pool2Node, 2, 2);
        }

        // ─────────────────────────────────────────────────────────────────
        // Layer 5: Fully Connected => 120
        // Each node sees 16×5×5=400 inputs
        // Per-node bias => 1 bias
        // ─────────────────────────────────────────────────────────────────
        layers[5] = new ConvolutionalNode[120];
        for (int i = 0; i < 120; i++) {
            // For fully connected, sizeZ=1, sizeY=1, sizeX=1 => 1 bias
            ConvolutionalNode fc1Node = new ConvolutionalNode(
                    5, i,
                    NodeType.HIDDEN,
                    activationType,
                    0,
                    batchSize,
                    1,
                    1,
                    1,
                    useDropout,
                    hiddenDropoutRate,
                    useBatchNormalization,
                    alpha
            );
            layers[5][i] = fc1Node;
            numberWeights += fc1Node.getNumberWeights(); // 1 bias

            // 400 weights per node => from each 16 maps × 5×5
            for (int j = 0; j < 16; j++) {
                new ConvolutionalEdge(layers[4][j], fc1Node, 1, 5, 5);
                numberWeights += 25;
            }
        }

        // ─────────────────────────────────────────────────────────────────
        // Layer 6: Fully Connected => 84
        // Each node sees 120 inputs => 120 weights + 1 bias
        // Dropout turned off
        // ─────────────────────────────────────────────────────────────────
        layers[6] = new ConvolutionalNode[84];
        for (int i = 0; i < 84; i++) {
            ConvolutionalNode fc2Node = new ConvolutionalNode(
                    6, i,
                    NodeType.HIDDEN,
                    activationType,
                    0,
                    batchSize,
                    1,
                    1,
                    1,
                    /*useDropout=*/false,
                    /*dropoutRate=*/0.0,
                    useBatchNormalization,
                    alpha
            );
            layers[6][i] = fc2Node;
            numberWeights += fc2Node.getNumberWeights(); // 1 bias

            // fully connect from 120 fc1 nodes => 120 weights
            for (int j = 0; j < 120; j++) {
                new ConvolutionalEdge(layers[5][j], fc2Node, 1, 1, 1);
                numberWeights += 1;
            }
        }

        // ─────────────────────────────────────────────────────────────────
        // Layer 7: OUTPUT => 10
        // No bias on output nodes
        // each node sees 84 inputs => 84 weights
        // ─────────────────────────────────────────────────────────────────
        layers[7] = new ConvolutionalNode[outputLayerSize];
        for (int i = 0; i < outputLayerSize; i++) {
            ConvolutionalNode outNode = new ConvolutionalNode(
                    7, i,
                    NodeType.OUTPUT,
                    activationType,
                    0,
                    batchSize,
                    1,
                    1,
                    1,
                    /*useDropout=*/false,
                    /*dropoutRate=*/0.0,
                    /*useBatchNormalization=*/false,
                    /*alpha=*/0.0
            );
            layers[7][i] = outNode;

            // fully connect from 84 fc2 nodes => 84 weights
            for (int j = 0; j < 84; j++) {
                new ConvolutionalEdge(layers[6][j], outNode, 1, 1, 1);
                numberWeights += 1;
            }
        }

    }

    public ConvolutionalNeuralNetwork(LossFunction lossFunction, boolean useDropout, double inputDropoutRate, double hiddenDropoutRate, boolean useBatchNormalization, double alpha) {
        this.lossFunction = lossFunction;
        this.useDropout = useDropout;
        this.inputDropoutRate = inputDropoutRate;
        this.hiddenDropoutRate = hiddenDropoutRate;
        this.useBatchNormalization = useBatchNormalization;
        this.alpha = alpha;
    }

    /**
     * This gets the number of weights in the ConvolutionalNeuralNetwork, which should
     * be equal to the number of hidden nodes (1 bias per hidden node) plus 
     * the number of edges (1 bias per edge). It is updated whenever an edge 
     * is added to the neural network.
     *
     * @return the number of weights in the neural network.
     */
    public int getNumberWeights() {
        return numberWeights;
    }

    /**
     * This resets all the values that are modified in the forward pass and 
     * backward pass and need to be reset to 0 before doing another
     * forward and backward pass (i.e., all the non-weights/biases).
     */
    public void reset() {
        for (int layer = 0; layer < layers.length; layer++) {
            for (int number = 0; number < layers[layer].length; number++) {
                //call reset on each node in the network
                layers[layer][number].reset();
            }
        }
    }

    /**
     * This resets the running averages for batch normalization
     * across all the nodes at the beginning of an epoch.
     */
    public void resetRunning() {
        for (int layer = 0; layer < layers.length; layer++) {
            for (int number = 0; number < layers[layer].length; number++) {
                //call reset on each node in the network
                layers[layer][number].resetRunning();
            }
        }
    }

    /**
     * This returns an array of every weight (including biases) in the ConvolutionalNeuralNetwork.
     * This will be very useful in backpropagation and sanity checking.
     *
     * @throws NeuralNetworkException if numberWeights was not calculated correctly. This shouldn't happen.
     */
    public double[] getWeights() throws NeuralNetworkException {
        double[] weights = new double[numberWeights];

        //What we're going to do here is fill in the weights array
        //we just created by having each node set the weights starting
        //at the position variable we're creating. The Node.getWeights
        //method will set the weights variable passed as a parameter,
        //and then return the number of weights it set. We can then
        //use this to increment position so the next node gets weights
        //and puts them in the right position in the weights array.
        int position = 0;
        for (int layer = 0; layer < layers.length; layer++) {
            for (int nodeNumber = 0; nodeNumber < layers[layer].length; nodeNumber++) {
                int nWeights = layers[layer][nodeNumber].getWeights(position, weights);
                position += nWeights;

                if (position > numberWeights) {
                    throw new NeuralNetworkException("The numberWeights field of the ConvolutionalNeuralNetwork was (" + numberWeights + ") but when getting the weights there were more hidden nodes and edges than numberWeights. This should not happen unless numberWeights is not being updated correctly.");
                }
            }
        }

        return weights;
    }

    /**
     * This sets every weight (including biases) in the ConvolutionalNeuralNetwork, it sets them in
     * the same order that they are retreived by the getWeights method.
     * This will be very useful in backpropagation and sanity checking. 
     *
     * @throws NeuralNetworkException if numberWeights was not calculated correctly. This shouldn't happen.
     */
    public void setWeights(double[] newWeights) throws NeuralNetworkException {
        if (numberWeights != newWeights.length) {
            throw new NeuralNetworkException("Could not setWeights because the number of new weights: " + newWeights.length + " was not equal to the number of weights in the ConvolutionalNeuralNetwork: " + numberWeights);
        }

        int position = 0;
        for (int layer = 0; layer < layers.length; layer++) {
            for (int nodeNumber = 0; nodeNumber < layers[layer].length; nodeNumber++) {
                Log.trace("setting weights for layer: " + layer + ", nodeNumber: " + nodeNumber + ", position: " + position);
                int nWeights = layers[layer][nodeNumber].setWeights(position, newWeights);
                position += nWeights;

                if (position > numberWeights) {
                    throw new NeuralNetworkException("The numberWeights field of the ConvolutionalNeuralNetwork was (" + numberWeights + ") but when setting the weights there were more hidden nodes and edges than numberWeights. This should not happen unless numberWeights is not being updated correctly.");
                }
            }
        }
    }

    /**
     * This returns an array of every weight (including biases) in the ConvolutionalNeuralNetwork.
     * This will be very useful in backpropagation and sanity checking.
     *
     * @throws NeuralNetworkException if numberWeights was not calculated correctly. This shouldn't happen.
     */
    public double[] getDeltas() throws NeuralNetworkException {
        double[] deltas = new double[numberWeights];

        //What we're going to do here is fill in the deltas array
        //we just created by having each node set the deltas starting
        //at the position variable we're creating. The Node.getDeltas
        //method will set the deltas variable passed as a parameter,
        //and then return the number of deltas it set. We can then
        //use this to increment position so the next node gets deltas
        //and puts them in the right position in the deltas array.
        int position = 0;
        for (int layer = 0; layer < layers.length; layer++) {
            for (int nodeNumber = 0; nodeNumber < layers[layer].length; nodeNumber++) {
                int nDeltas = layers[layer][nodeNumber].getDeltas(position, deltas);
                position += nDeltas;

                if (position > numberWeights) {
                    throw new NeuralNetworkException("The numberWeights field of the ConvolutionalNeuralNetwork was (" + numberWeights + ") but when getting the deltas there were more hidden nodes and edges than numberWeights. This should not happen unless numberWeights is not being updated correctly.");
                }
            }
        }

        return deltas;
    }


    /**
     * This initializes the weights in the RNN using either Xavier or
     * Kaiming initialization.
    *
     * @param type will be either "xavier" or "kaiming" and this will
     * initialize the child nodes accordingly, using their helper methods.
     * @param bias is the value to set the bias of each node to.
     */
    public void initializeRandomly(String type, double bias) {
        //TODO: You need to implement this for Programming Assignment 3 - Part 1

        for (int layerIndex = 0; layerIndex < layers.length; layerIndex++) {
            for (int nodeNumber = 0; nodeNumber < layers[layerIndex].length; nodeNumber++) {
                ConvolutionalNode node = layers[layerIndex][nodeNumber];

                // Typically, we only initialize biases on HIDDEN nodes.
                // The assignment notes that OUTPUT nodes have no bias, and
                // INPUT nodes don't learn weights. So skip those if you want.
                if (node.nodeType == NodeType.HIDDEN) {

                    // ─── 1) Compute fanIn/fanOut from the edges feeding in/out of this node ───
                    int fanIn = 0;
                    for (Edge e : node.inputEdges) {
                        // Each Edge has sizeZ, sizeY, sizeX for its filter or pooling region
                        fanIn += (e.sizeZ * e.sizeY * e.sizeX);
                    }
                    // Just to be safe – if there's no input edge, avoid dividing by zero:
                    if (fanIn < 1) {
                        fanIn = 1;
                    }

                    // For Xavier, we usually also compute fanOut:
                    int fanOut = 0;
                    for (Edge e : node.outputEdges) {
                        fanOut += (e.sizeZ * e.sizeY * e.sizeX);
                    }
                    if (fanOut < 1) {
                        fanOut = 1;
                    }

                    // ─── 2) Call the appropriate initialization method on the node ───
                    if ("kaiming".equalsIgnoreCase(type)) {
                        node.initializeWeightsAndBiasKaiming(bias, fanIn);
                    } else if ("xavier".equalsIgnoreCase(type)) {
                        node.initializeWeightsAndBiasXavier(bias, fanIn, fanOut);
                    } else {
                        System.err.println("Unknown initialization type: " + type);
                    }
                }
            }
        }
    }



    /**
     * This performs a forward pass through the neural network given
     * inputs from the input instance.
     *
     * @param imageDataSet is the imageDataSet we're using to train the CNN
     * @param startIndex is the index of the first image to use
     * @param batchSize is the number of images to use
     *
     * @return the sum of the output of all output nodes
     */
    public double forwardPass(ImageDataSet imageDataSet, int startIndex, int batchSize, boolean training) throws NeuralNetworkException {
        //be sure to reset before doing a forward pass
        reset();

        //set input values differently for time series and character sequences

        //set the input nodes for each time step in the CharacterSequence

        List<Image> images = imageDataSet.getImages(startIndex, batchSize);

        for (int number = 0; number < layers[0].length; number++) {
            ConvolutionalNode inputNode = layers[0][number];
            inputNode.setValues(images, imageDataSet.getChannelAvgs(), imageDataSet.getChannelStdDevs(imageDataSet.getChannelAvgs()));
        }



        for (int layerIndex = 0; layerIndex < layers.length; layerIndex++) {
            for (int nodeNumber = 0; nodeNumber < layers[layerIndex].length; nodeNumber++) {
                layers[layerIndex][nodeNumber].propagateForward(training);
            }
        }


        //The following is needed for Programming Assignment 2 - Part 1
        int outputLayer = layers.length - 1;
        int nOutputs = layers[outputLayer].length;


        //note that the target value for any time step is the sequence value at that time step + 1
        //this means you should only go up to length - 1 time steps in calculating the loss
        double lossSum = 0;

        if (lossFunction == LossFunction.SVM) {

            //to calculate for each image in the batch
            for (int i = 0; i < batchSize; i++) {
                int correctLabel = images.get(i).label;
                double correctScore = layers[outputLayer][correctLabel].outputValues[i][0][0][0];

                // Initialize all output node deltas to zero for this example.
                for (int j = 0; j < nOutputs; j++) {
                    layers[outputLayer][j].delta[i][0][0][0] = 0.0;
                }

                int violationsCount = 0;
                // For each output node, if the margin is positive, add the appropriate delta.
                for (int j = 0; j < nOutputs; j++) {
                    if (j == correctLabel) continue;
                    double score_j = layers[outputLayer][j].outputValues[i][0][0][0];
                    double margin = 1.0 + score_j - correctScore;
                    if (margin > 0.0) {
                        lossSum += margin;
                        layers[outputLayer][j].delta[i][0][0][0] = 1.0;
                        violationsCount++;
                    }
                }
                // For the correct label, subtract the total number of margin violations.
                layers[outputLayer][correctLabel].delta[i][0][0][0] = -violationsCount;
            }


        } else if (lossFunction == LossFunction.SOFTMAX) {

            //to calculate for each image in the batch
            // For each image in the batch
            for (int i = 0; i < batchSize; i++) {
                int label = images.get(i).label;
                double[] logits = new double[nOutputs];

                // Gather logits from each output node
                for (int j = 0; j < nOutputs; j++) {
                    logits[j] = layers[outputLayer][j].outputValues[i][0][0][0];
                }

                // Find maximum for numerical stability
                double maxLogit = Double.NEGATIVE_INFINITY;
                for (int j = 0; j < nOutputs; j++) {
                    if (logits[j] > maxLogit) {
                        maxLogit = logits[j];
                    }
                }

                // Compute sum of exponentials
                double sumExp = 0.0;
                for (int j = 0; j < nOutputs; j++) {
                    sumExp += Math.exp(logits[j] - maxLogit);
                }

                // Safety check for numerical stability
                if (sumExp < 1e-10) sumExp = 1e-10;

                // Compute probabilities and loss, and assign deltas
                double probCorrect = Math.exp(logits[label] - maxLogit) / sumExp;

                // Safety check to prevent log(0)
                probCorrect = Math.max(1e-15, Math.min(1.0 - 1e-15, probCorrect));

                for (int j = 0; j < nOutputs; j++) {
                    double p = Math.exp(logits[j] - maxLogit) / sumExp;
                    // For softmax cross-entropy, the derivative is p - 1 for the correct label, and p for others
                    if (j == label) {
                        layers[outputLayer][j].delta[i][0][0][0] = p - 1.0;
                    } else {
                        layers[outputLayer][j].delta[i][0][0][0] = p;
                    }
                }

                double loss = -Math.log(probCorrect);
                // Safety check to prevent infinite loss
                if (Double.isInfinite(loss) || Double.isNaN(loss)) {
                    loss = 10.0; // Reasonable upper bound
                }
                lossSum += loss;
            }
        } else {
            throw new NeuralNetworkException("Could not do a CharacterSequence forward pass on ConvolutionalNeuralNetwork because lossFunction was unknown or invalid: " + lossFunction);
        }

        return lossSum;
    }

    /**
     * This does forward passes over the entire image data set to calculate
     * the total error and accuracy (this is used by GradientDescent.java). We
     * do them both here to improve performance.
     *
     * @param imageDataSet is the imageDataSet we're using to train the CNN
     * @param accuracyAndError is a double array of length 2, index 0 will
     * be the accuracy and index 1 will be the error
     */
    public void calculateAccuracyAndError(ImageDataSet imageDataSet, int batchSize, double[] accuracyAndError) throws NeuralNetworkException {

        //the output node with the maximum value is the predicted class
        //you need to sum up how many of these match the actual class
        //for each time step of each sequence, and then calculate: 
        //num correct / total
        //to get a percentage accuracy

        double lossSum = forwardPass(imageDataSet, 0, batchSize, false);
        List<Image> images = imageDataSet.getImages(0, batchSize);

        int lastLayer = layers.length - 1;
        int nOutputs = layers[lastLayer].length;
        int correct = 0;

        // For each image in the batch, compute the predicted class.
        for (int i = 0; i < batchSize; i++) {
            double maxVal = -Double.MAX_VALUE;
            int pred = -1;
            for (int j = 0; j < nOutputs; j++) {
                // The output value is stored at outputValues[i][0][0][0] for each output node.
                double val = layers[lastLayer][j].outputValues[i][0][0][0];
                if (val > maxVal) {
                    maxVal = val;
                    pred = j;
                }
            }
            // Compare predicted label with the actual label.
            if (pred == images.get(i).label) {
                correct++;
            }
        }

        double accuracy = ((double) correct) / batchSize;
        double avgLoss = lossSum / batchSize;
        accuracyAndError[0] = accuracy;
        accuracyAndError[1] = avgLoss;

    }


    /**
     * This gets the output values of the neural network 
     * after a forward pass, this will be a 1 dimensional array, one
     * value for each output node
     *
     * @param batchSize is the batch size of for this CNN
     *
     * @return a one dimensional array of the output values from this neural network for
     */
    public double[][] getOutputValues(int batchSize) {
        //the number of output values is the number of output nodes
        int outputLayer = layers.length - 1;
        int nOutputs = layers[outputLayer].length;

        double[][] outputValues = new double[batchSize][nOutputs];

        for (int i = 0; i < batchSize; i++) {
            for (int number = 0; number < nOutputs; number++) {
                outputValues[i][number] = layers[outputLayer][number].outputValues[i][0][0][0];
            }
        }

        return outputValues;
    }

    /**
     * The step size used to calculate the gradient numerically using the finite
     * difference method.
     */
    private static final double H = 0.0000001;

    /**
     * This calculates the gradient of the neural network with it's current
     * weights for a given DataSet Instance using the finite difference method:
     * gradient[i] = (f(x where x[i] = x[i] + H) - f(x where x[i] = x[i] - H)) / 2h
     *
     * @param imageDataSet is the imageDataSet we're using to train the CNN
     * @param startIndex is the index of the first image to use
     * @param batchSize is the number of images to use
     */
    public double[] getNumericGradient(ImageDataSet imageDataSet, int startIndex, int batchSize) throws NeuralNetworkException {

        // Save the current weights
        double[] originalWeights = getWeights();
        double[] numericGradient = new double[numberWeights];

        // For each weight, compute the numerical gradient
        for (int i = 0; i < numberWeights; i++) {
            double temp = originalWeights[i];

            // Perturb positively
            originalWeights[i] = temp + H;
            setWeights(originalWeights);
            reset(); // Reset all non-weight values
            double lossPlus = forwardPass(imageDataSet, startIndex, batchSize, true); // Use training=true

            // Perturb negatively
            originalWeights[i] = temp - H;
            setWeights(originalWeights);
            reset(); // Reset all non-weight values
            double lossMinus = forwardPass(imageDataSet, startIndex, batchSize, true); // Use training=true

            // Calculate gradient approximation
            numericGradient[i] = (lossPlus - lossMinus) / (2 * H);

            // Restore original weight
            originalWeights[i] = temp;
        }

        // Restore the original weights in the network
        setWeights(originalWeights);
        return numericGradient;
    }


    /**
     * This performs a backward pass through the neural network given 
     * outputs from the given instance. This will set the deltas in
     * all the edges and nodes which will be used to calculate the 
     * gradient and perform backpropagation.
     *
     * @param imageDataSet is the imageDataSet we're using to train the CNN
     * @param startIndex is the index of the first image to use
     * @param batchSize is the number of images to use
     *
     */
    public void backwardPass(ImageDataSet imageDataSet, int startIndex, int batchSize) throws NeuralNetworkException {


        // Now simply backprop from the last layer to the first:
        for (int layerIndex = layers.length - 1; layerIndex >= 0; layerIndex--) {
            for (int nodeNumber = 0; nodeNumber < layers[layerIndex].length; nodeNumber++) {
                // Each ConvolutionalNode is responsible for distributing its delta
                // backward to input edges and accumulating the partial derivatives
                // w.r.t. weights (and bias). Typically there's a method like:
                layers[layerIndex][nodeNumber].propagateBackward();
            }
        }

    }

    /**
     * This gets the gradient of the neural network at its current
     * weights and the given instance using backpropagation (e.g., 
     * the ConvolutionalNeuralNetwork.backwardPass(Sequence)) Method.
     *
     * Helpful tip: use getDeltas after doing the propagateBackwards through
     * the networks to get the gradients/deltas in the same order as the
     * weights (which will be the same order as they're calculated for
     * the numeric gradient).
     *
     * @param imageDataSet is the imageDataSet we're using to train the CNN
     * @param startIndex is the index of the first image to use
     * @param batchSize is the number of images to use
     */
    public double[] getGradient(ImageDataSet imageDataSet, int startIndex, int batchSize) throws NeuralNetworkException {
        forwardPass(imageDataSet, startIndex, batchSize, true /*we're training here so use the training versions of batch norm and dropout*/);
        backwardPass(imageDataSet, startIndex, batchSize);

        return getDeltas();
    }

    /**
     * Print out numeric vs backprop gradients in a clean manner so that
     * you can see where gradients were not the same
     *
     * @param numericGradient is a previously calculated numeric gradient
     * @param backpropGradient is a previously calculated gradient from backprop
     */
    public void printGradients(double[] numericGradient, double[] backpropGradient) {
        int current = 0;
        for (int layer = 0; layer < layers.length; layer++) {
            for (int number = 0; number < layers[layer].length; number++) {
                //call reset on each node in the network
                current += layers[layer][number].printGradients(current, numericGradient, backpropGradient);
            }
        }
    }
}
