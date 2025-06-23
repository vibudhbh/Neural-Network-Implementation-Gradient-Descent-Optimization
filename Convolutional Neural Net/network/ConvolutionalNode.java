/**
 * This class represents a Node in the neural network. It will
 * have a list of all input and output edges, as well as its
 * own value. It will also track it's layer in the network and
 * if it is an input, hidden or output node.
 */
package network;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import data.Image;

import util.Log;

public class ConvolutionalNode {
    //the layer of this node in the neural network. This is set 
    //final so we cannot change it after assigning it (which 
    //could cause bugs)
    public final int layer;

    //the number of this node in it's layer. this is mostly
    //just used for printing friendly error message
    public final int number;

    //padding for the image (usually only used on the input nodes)
    public final int padding;

    //the number of images in each batch
    public final int batchSize;

    //the size of the Z dimension of this feature map
    public final int sizeZ;

    //the size of the Y dimension of this feature map
    public final int sizeY;

    //the size of the X dimension of this feature map
    public final int sizeX;

    //specifies if this node will use dropout
    public final boolean useDropout;

    //specifies the dropout rate for this node
    public final double dropoutRate;

    //specifies if the node will use batch normalization
    public final boolean useBatchNormalization;

    //the type of this node (input, hidden or output). This 
    //is set final so we cannot change it after assigning it 
    //(which could cause bugs)
    public final NodeType nodeType;

    //the activation function this node will use, can be
    //either sigmoid, tanh or softmax (for the output
    //layer).
    public final ActivationType activationType;

    //these are the values which are calculated by the forward
    //pass (if it is hidden or output), or assigned by the
    //data set if it is an input node, before the activation
    //function, dropout, batch normalization and pooling is applied
    public double[][][][] inputValues;

    //this is the value which is calculated by the forward
    //pass (if it is not an input node) after the activation
    //function has been applied
    public double[][][][] outputValues;

    //this is the delta/error calculated by backpropagation
    public double[][][][] delta;

    //this is the bias value added to the sum of the inputs
    //multiplied by the weights before the activation function
    //is applied
    protected double[][][] bias;

    //thius is the delta/error calculated by backpropagation
    //for the bias
    protected double[][][][] biasDelta;

    //this is a list of all incoming edges to this node
    protected List<Edge> inputEdges;

    //this is a list of all outgoing edges from this node
    protected List<Edge> outputEdges;


    //we're going to use this for dropout so we can re-create
    //dropout instances so we can unit test it
    public Random generator;


    //set these values for the deltas to backpropagate through dropout
    public double[][][][] dropoutDelta;


    //this is the beta value for batch normalization
    public double beta;
    //this is the delta for beta calculated on the
    //backwards pass for batch normalization
    public double betaDelta;

    //this is the gamma value for batch normalization
    public double gamma;
    //this is the delta for gamma calculated on the
    //backwards pass for batch normalization
    public double gammaDelta;

    //these are used by batch normalization to calculate the values
    //before multiplying by gamma and adding delta
    public double[][][][] xHat;

    //these are used by batch normalization to calculate the values
    //before multiplying by gamma and adding delta
    public double[][][][] afterBatchNorm;

    //this is used for calculating square root of sigma squared
    //in batch normalization
    double epsilon = 1e-7;

    //save these on the forward pass so we can reuse them on the
    //backward pass
    public double mu_b;
    public double sigma2_b;

    //can use these to test batch normalization backprop
    public double mu_delta;
    public double sigma2_delta;

    //this is used for calculating the batch normalization running
    //averages
    double alpha;

    //use these for the inference (testing) pass, calculate them
    //during the training pass
    double sigma2_running;
    double mu_running;

    /**
     * This creates a new node at a given layer in the
     * network and specifies it's type (either input,
     * hidden, our output).
     *
     * @param layer is the layer of the Node in
     * the neural network
     * @param nodeType is the type of node, specified by
     * the Node.NodeType enumeration.
     */
    public ConvolutionalNode(int layer, int number, NodeType nodeType, ActivationType activationType, int padding, int batchSize, int sizeZ, int sizeY, int sizeX, boolean useDropout, double dropoutRate, boolean useBatchNormalization, double alpha) {
        sizeY = sizeY + (2 * padding);
        sizeX = sizeX + (2 * padding);

        this.layer = layer;
        this.number = number;
        this.nodeType = nodeType;
        this.activationType = activationType;
        this.padding = padding;
        this.batchSize = batchSize;
        this.sizeZ = sizeZ;
        this.sizeY = sizeY;
        this.sizeX = sizeX;
        this.useDropout = useDropout;
        this.dropoutRate = dropoutRate;
        this.useBatchNormalization = useBatchNormalization;
        this.alpha = alpha;

        inputValues = new double[batchSize][sizeZ][sizeY][sizeX];
        outputValues = new double[batchSize][sizeZ][sizeY][sizeX];
        delta = new double[batchSize][sizeZ][sizeY][sizeX];
        bias = new double[sizeZ][sizeY][sizeX];
        biasDelta = new double[batchSize][sizeZ][sizeY][sizeX];
        dropoutDelta = new double[batchSize][sizeZ][sizeY][sizeX];
        xHat = new double[batchSize][sizeZ][sizeY][sizeX];
        afterBatchNorm = new double[batchSize][sizeZ][sizeY][sizeX];

        //initialize the input and output edges lists
        //as ArrayLists
        inputEdges = new ArrayList<Edge>();
        outputEdges = new ArrayList<Edge>();

        beta = 0.0;
        gamma = 1.0;
        Log.trace("Created a node: " + toString());

        sigma2_running = 1;
        mu_running = 0;

        generator = new Random();
    }

    /**
     * This resets the values which need to be recalcualted for
     * each forward and backward pass. It will also reset the
     * deltas for outgoing nodes.
     */
    public void reset() {
        Log.trace("Resetting node: " + toString());
        Log.trace("inputValues.length: " + inputValues.length);
        Log.trace("inputValues[0].length: " + inputValues[0].length);
        Log.trace("inputValues[0][0].length: " + inputValues[0][0].length);
        Log.trace("inputValues[0][0][0].length: " + inputValues[0][0][0].length);

        for (int i = 0; i < batchSize; i++) {
            for (int z = 0; z < sizeZ; z++) {
                for (int y = 0; y < sizeY; y++) {
                    for (int x = 0; x < sizeX; x++) {
                        inputValues[i][z][y][x] = 0;
                        outputValues[i][z][y][x] = 0;
                        delta[i][z][y][x] = 0;
                        biasDelta[i][z][y][x] = 0;
                        xHat[i][z][y][x] = 0;
                        afterBatchNorm[i][z][y][x] = 0;
                        dropoutDelta[i][z][y][x] = 0.0;
                    }
                }
            }
        } 
        betaDelta = 0.0;
        gammaDelta = 0.0;

        for (Edge outputEdge : outputEdges) {
            outputEdge.reset();
        }
    }

    /**
     * This resets the batch normalization running
     * statistics. It should only be done at the
     * beginning of an epoch
     */
    public void resetRunning() {
        sigma2_running = 1;
        mu_running = 0;
    }


    /**
     * The Edge class will call this on its output Node when it is
     * constructed (the input and output nodes of an edge
     * are passed as parameters to the Edge constructor).
     *
     * @param outgoingEdge the new outgoingEdge to add
     *
     * @throws NeuralNetworkException if the edge already exists
     */
    public void addOutgoingEdge(Edge outgoingEdge) throws NeuralNetworkException {
        //lets have a sanity check to make sure we don't duplicate adding
        //an edge
        for (Edge edge : outputEdges) {
            if (edge.equals(outgoingEdge)) {
                throw new NeuralNetworkException("Attempted to add an outgoing edge to node " + toString()
                        + " but could not as it already had an edge to the same output node: " + edge.outputNode.toString());
            }
        }

        Log.trace("Node " + toString() + " added outgoing edge to Node " + outgoingEdge.outputNode);
        outputEdges.add(outgoingEdge);
    }

    /**
     * The Edge class will call this on its input Node when it is
     * constructed (the input and output nodes of an edge
     * are passed as parameters to the Edge constructor).
     *
     * @param incomingEdge the new incomingEdge to add
     *
     * @throws NeuralNetworkException if the edge already exists
     */
    public void addIncomingEdge(Edge incomingEdge) throws NeuralNetworkException {
        //lets have a sanity check to make sure we don't duplicate adding
        //an edge
        for (Edge edge : inputEdges) {
            if (edge.equals(incomingEdge)) {
                throw new NeuralNetworkException("Attempted to add an incoming edge to node " + toString()
                        + " but could not as it already had an edge to the same input node: " + edge.inputNode.toString());
            }
        }

        Log.trace("Node " + toString() + " added incoming edge from Node " + incomingEdge.inputNode);
        inputEdges.add(incomingEdge);
    }

    public void setValues(List<Image> images, double[] avgs, double[] stdDevs) throws NeuralNetworkException {
        if (images.size() != batchSize) {
            throw new NeuralNetworkException("Cannot set ConvolutioanlNode values because node's batchSize (" + batchSize +") != number of images (" + images.size() + ")");
        }

        for (int i = 0; i < images.size(); i++) {
            Image image = images.get(i);
            for (int z = 0; z < sizeZ; z++) {
                for (int y = 0; y < sizeY - (2 * padding); y++) {
                    for (int x = 0; x < sizeX - (2 * padding); x++) {
                        outputValues[i][z][y + padding][x + padding] = (Byte.toUnsignedInt(image.pixels[z][y][x]) - avgs[z]) / stdDevs[z];
                    }
                }
            }
        }
    }

    /**
     * Used to print gradients related to this node, along with informationa
     * about this node.
     * It start printing the gradients passed in starting at position, and 
     * return the number of gradients it printed.
     *
     * @param position is the index to start printing different gradients
     * @param numericGradient is the array of the numeric gradient we're printing
     * @param backpropGradient is the array of the backprop gradient we're printing
     *
     * @return the number of gradients printed by this node (and its outgoing edges)
     */
    public int printGradients(int position, double[] numericGradient, double[] backpropGradient) {
        int count = 0;

        Log.info("ConvolutionalNode [layer: " + layer + ", number: " + number + ", batchSize: " + batchSize + "]:");
        //the first weight set will be the bias if it is a hidden node
        if (nodeType == NodeType.HIDDEN) {
            for (int z = 0; z < sizeZ; z++) {
                for (int y = 0; y < sizeY; y++) {
                    for (int x = 0; x < sizeX; x++) {
                        Log.info("\tbias[" + z + "][" + y + "][" + x + "]: " + Log.twoGradients(numericGradient[position + count], backpropGradient[position + count]));
                        count++;
                    }
                }
            }
        }

        if (useBatchNormalization) {
            Log.info("\tbeta:  " + Log.twoGradients(numericGradient[position + count], backpropGradient[position + count]));
            count++;
            Log.info("\tgamma:  " + Log.twoGradients(numericGradient[position + count], backpropGradient[position + count]));
            count++;
        }

        for (Edge edge : outputEdges) {
            count += edge.printGradients(position + count, numericGradient, backpropGradient);
        }

        return count;
    }


    /**
     * Used to get the weights of this node along with the weights
     * of all of it's outgoing edges. It will set the weights in the weights
     * parameter passed in starting at position, and return the number of
     * weights it set.
     *
     * @param position is the index to start setting weights in the weights parameter
     * @param weights is the array of weights we're setting.
     *
     * @return the number of weights set in the weights parameter
     */
    public int getWeights(int position, double[] weights) {
        int weightCount = 0;

        //the first weight set will be the bias if it is a hidden node
        if (nodeType == NodeType.HIDDEN) {
            for (int z = 0; z < sizeZ; z++) {
                for (int y = 0; y < sizeY; y++) {
                    for (int x = 0; x < sizeX; x++) {
                        weights[position + weightCount] = bias[z][y][x];
                        weightCount++;
                    }
                }
            }
        }

        if (useBatchNormalization) {
            weights[position + weightCount] = beta;
            weightCount++;
            weights[position + weightCount] = gamma;
            weightCount++;
        }

        for (Edge edge : outputEdges) {
            weightCount += edge.getWeights(position + weightCount, weights);
        }

        return weightCount;
    }

    /**
     * Used to get the deltas of this node along with the deltas
     * of all of it's outgoing edges. It will set the deltas in the deltas
     * parameter passed in starting at position, and return the number of
     * deltas it set.
     *
     * @param position is the index to start setting deltas in the deltas parameter
     * @param deltas is the array of deltas we're setting.
     *
     * @return the number of deltas set in the deltas parameter
     */
    public int getDeltas(int position, double[] deltas) {
        int deltaCount = 0;

        //the first delta set will be the bias if it is a hidden node
        if (nodeType == NodeType.HIDDEN) {
            for (int z = 0; z < sizeZ; z++) {
                for (int y = 0; y < sizeY; y++) {
                    for (int x = 0; x < sizeX; x++) {
                        deltas[position + deltaCount] = 0;

                        for (int i = 0; i < batchSize; i++) {
                            deltas[position + deltaCount] += biasDelta[i][z][y][x];
                        }

                        deltaCount++;
                    }
                }
            }

            if (useBatchNormalization) {
                deltas[position + deltaCount] = betaDelta;
                deltaCount++;
                deltas[position + deltaCount] = gammaDelta;
                deltaCount++;
            }

        }

        for (Edge edge : outputEdges) {
            deltaCount += edge.getDeltas(position + deltaCount, deltas);
        }

        return deltaCount;
    }
    
    /**
     * Get the number of weights in this node. Note that this will
     * be different if it is using batch normalization or not (if it is
     * then it will also have an additional gamma and beta weights)
     *
     * @return the number of weights this node uses
     */
    public int getNumberWeights() {
        if (nodeType == NodeType.HIDDEN) {
            int n = sizeZ * sizeY * sizeX;
            if (useBatchNormalization) n += 2;
            return n;
        } else {
            return 0;
        }
    }


    /**
     * Used to set the weights of this node along with the weights of
     * all it's outgoing edges. It uses the same technique as Node.getWeights
     * where the starting position of weights to set is passed, and it returns
     * how many weights were set.
     * 
     * @param position is the starting position in the weights parameter to start
     * setting weights from.
     * @param weights is the array of weights we are setting from
     *
     * @return the number of weights gotten from the weights parameter
     */
    public int setWeights(int position, double[] weights) {
        int weightCount = 0;

        //the first weight set will be the bias if it is a hidden node
        if (nodeType == NodeType.HIDDEN) {
            for (int z = 0; z < sizeZ; z++) {
                for (int y = 0; y < sizeY; y++) {
                    for (int x = 0; x < sizeX; x++) {
                        bias[z][y][x] = weights[position + weightCount];
                        weightCount++;
                    }
                }
            }

            if (useBatchNormalization) {
                beta = weights[position + weightCount];
                weightCount++;
                gamma = weights[position + weightCount];
                weightCount++;
            }

        }

        for (Edge edge : outputEdges) {
            weightCount += edge.setWeights(position + weightCount, weights);
        }

        return weightCount;
    }


    /**
     * This propagates the outputValues at this node
     * to all it's output nodes.
     */
    public void propagateForward(boolean training) {


        if (nodeType == NodeType.HIDDEN) {
            //batch normalization happens after the activation function but before dropout

            if (useBatchNormalization) {
                //TODO: Implement this for Programming Assignment 3 - Part 3

                if (training) {
                    // 1) Calculate mean across the entire batch
                    mu_b = 0.0;
                    for (int i = 0; i < batchSize; i++) {
                        for (int z = 0; z < sizeZ; z++) {
                            for (int y = 0; y < sizeY; y++) {
                                for (int x = 0; x < sizeX; x++) {
                                    mu_b += inputValues[i][z][y][x];
                                }
                            }
                        }
                    }
                    mu_b /= (batchSize * sizeZ * sizeY * sizeX);

                    // 2) Calculate variance
                    sigma2_b = 0.0;
                    for (int i = 0; i < batchSize; i++) {
                        for (int z = 0; z < sizeZ; z++) {
                            for (int y = 0; y < sizeY; y++) {
                                for (int x = 0; x < sizeX; x++) {
                                    double diff = inputValues[i][z][y][x] - mu_b;
                                    sigma2_b += diff * diff;
                                }
                            }
                        }
                    }
                    sigma2_b /= (batchSize * sizeZ * sizeY * sizeX);

                    // 3) Update running averages (for inference) - see slide 24
                    mu_running = alpha * mu_b + (1.0 - alpha) * mu_running;
                    sigma2_running = alpha * sigma2_b + (1.0 - alpha) * sigma2_running;

                    // 4) Calculate x-hat (normalize values)
                    for (int i = 0; i < batchSize; i++) {
                        for (int z = 0; z < sizeZ; z++) {
                            for (int y = 0; y < sizeY; y++) {
                                for (int x = 0; x < sizeX; x++) {
                                    xHat[i][z][y][x] = (inputValues[i][z][y][x] - mu_b) / Math.sqrt(sigma2_b + epsilon);
                                }
                            }
                        }
                    }

                    // 5) Scale and shift with gamma and beta
                    for (int i = 0; i < batchSize; i++) {
                        for (int z = 0; z < sizeZ; z++) {
                            for (int y = 0; y < sizeY; y++) {
                                for (int x = 0; x < sizeX; x++) {
                                    afterBatchNorm[i][z][y][x] = gamma * xHat[i][z][y][x] + beta;
                                }
                            }
                        }
                    }
                } else {
                    // During inference, use running averages (see slide 24)
                    for (int i = 0; i < batchSize; i++) {
                        for (int z = 0; z < sizeZ; z++) {
                            for (int y = 0; y < sizeY; y++) {
                                for (int x = 0; x < sizeX; x++) {
                                    xHat[i][z][y][x] = (inputValues[i][z][y][x] - mu_running) / Math.sqrt(sigma2_running + epsilon);
                                    afterBatchNorm[i][z][y][x] = gamma * xHat[i][z][y][x] + beta;
                                }
                            }
                        }
                    }
                }
            }

            //don't forget to use the outputs of batchnorm as the
            //inputs to your activation functions (as opposed to
            //just the inputs)
            double[][][][] inputs;
            if (useBatchNormalization) {
                inputs = afterBatchNorm;
            } else {
                inputs = inputValues;
            }

            if (activationType == ActivationType.NONE) {
                for (int i = 0; i < batchSize; i++) {
                    for (int z = 0; z < sizeZ; z++) {
                        for (int y = 0; y < sizeY; y++) {
                            for (int x = 0; x < sizeX; x++) {
                                outputValues[i][z][y][x] = inputs[i][z][y][x] + bias[z][y][x];
                            }
                        }
                    }
                }

            } else if (activationType == ActivationType.RELU) {


                for (int i = 0; i < batchSize; i++) {
                    for (int z = 0; z < sizeZ; z++) {
                        for (int y = 0; y < sizeY; y++) {
                            for (int x = 0; x < sizeX; x++) {
                                double v = inputs[i][z][y][x] + bias[z][y][x];
                                outputValues[i][z][y][x] = (v > 0.0) ? v : 0.0;
                            }
                        }
                    }
                }

            } else if (activationType == ActivationType.RELU5) {


                for (int i = 0; i < batchSize; i++) {
                    for (int z = 0; z < sizeZ; z++) {
                        for (int y = 0; y < sizeY; y++) {
                            for (int x = 0; x < sizeX; x++) {
                                double v = inputs[i][z][y][x] + bias[z][y][x];
                                if (v < 0.0) {
                                    v = 0.0;
                                } else if (v > 5.0) {
                                    v = 5.0;
                                }
                                outputValues[i][z][y][x] = v;
                            }
                        }
                    }
                }

            } else if (activationType == ActivationType.LEAKY_RELU5) {

                double leak = 0.01;
                for (int i = 0; i < batchSize; i++) {
                    for (int z = 0; z < sizeZ; z++) {
                        for (int y = 0; y < sizeY; y++) {
                            for (int x = 0; x < sizeX; x++) {
                                double v = inputs[i][z][y][x] + bias[z][y][x];
                                if (v < 0.0) {
                                    v = leak * v;
                                } else if (v >= 5.0) {
                                    v = 5.0;           // inclusive boundary
                                }
                                outputValues[i][z][y][x] = v;
                            }
                        }
                    }
                }
            }

        } else if (nodeType == NodeType.OUTPUT) {
            //no batch norm or bias on the output layer

            for (int i = 0; i < batchSize; i++) {
                for (int z = 0; z < sizeZ; z++) {
                    for (int y = 0; y < sizeY; y++) {
                        for (int x = 0; x < sizeX; x++) {
                            double inputValue = inputValues[i][z][y][x];
                            outputValues[i][z][y][x] = inputValue;
                        }
                    }
                }
            }
        }

        //dropout happens after the activation function
        if (nodeType == NodeType.HIDDEN || nodeType == NodeType.INPUT) {
            //we can use dropout on either hidden or input nodes
            if (useDropout) {
                //TODO: Implement this for Programming Assignment 3 - Part 3
                //NOTE: please use generator.nextDouble() to deterine if a
                //cell is used or not -- the unit tests to check correctness
                //will not work if you use Math.random().

                if (training) {
                    // Training version - inverted dropout (slides page 11)
                    double keepProb = 1.0 - dropoutRate;
                    double scale = 1.0 / keepProb;

                    for (int i = 0; i < batchSize; i++) {
                        for (int z = 0; z < sizeZ; z++) {
                            for (int y = 0; y < sizeY; y++) {
                                for (int x = 0; x < sizeX; x++) {
                                    // Generate random value
                                    double rand = generator.nextDouble();

                                    // Keep node with probability (1 - dropoutRate)
                                    if (rand < keepProb) {
                                        // Keep this neuron and scale up by 1/keepProb
                                        dropoutDelta[i][z][y][x] = scale;
                                        outputValues[i][z][y][x] *= scale;
                                    } else {
                                        // Drop this neuron
                                        dropoutDelta[i][z][y][x] = 0.0;
                                        outputValues[i][z][y][x] = 0.0;
                                    }
                                }
                            }
                        }
                    }

                } else {
                    //do inference version of dropout
                    //since we're doing inverted dropout this
                    //doesn't need to do anything
                    // For inference, with inverted dropout, we don't need to do anything
                    // Just ensure dropoutDelta is properly initialized
                    for (int i = 0; i < batchSize; i++) {
                        for (int z = 0; z < sizeZ; z++) {
                            for (int y = 0; y < sizeY; y++) {
                                for (int x = 0; x < sizeX; x++) {
                                    dropoutDelta[i][z][y][x] = 1.0; // No scaling during inference
                                }
                            }
                        }
                    }
                }
            }
        }

        for (Edge outputEdge : outputEdges) {
            outputEdge.propagateForward(outputValues);
        }
    }

    /**
     * This propagates the delta back from this node
     * to its incoming edges.
     */
    public void propagateBackward() {


        if (nodeType == NodeType.HIDDEN || nodeType == NodeType.INPUT) {

            if (useDropout) {
                //TODO: Implement this for Programming Assignment 3 - Part 3

                for (int i = 0; i < batchSize; i++) {
                    for (int z = 0; z < sizeZ; z++) {
                        for (int y = 0; y < sizeY; y++) {
                            for (int x = 0; x < sizeX; x++) {
                                // Multiply deltas by the dropout mask (slides page 9)
                                delta[i][z][y][x] *= dropoutDelta[i][z][y][x];
                            }
                        }
                    }
                }
            }
        }

        if (nodeType == NodeType.HIDDEN) {
            //don't forget to use the outputs of batchnorm as the
            //inputs to your activation functions (as opposed to
            //just the inputs)
            double[][][][] inputs;
            if (useBatchNormalization) {
                inputs = afterBatchNorm;
            } else {
                inputs = inputValues;
            }


            if (activationType == ActivationType.NONE) {
                //don't need to do change the deltas but
                //still calculate bias deltas

                for (int i = 0; i < batchSize; i++) {
                    for (int z = 0; z < sizeZ; z++) {
                        for (int y = 0; y < sizeY; y++) {
                            for (int x = 0; x < sizeX; x++) {
                                biasDelta[i][z][y][x] += delta[i][z][y][x];
                            }
                        }
                    }
                }

            } else if (activationType == ActivationType.RELU) {

                // For RELU: derivative is 1 if v > 0, else 0.
                for (int i = 0; i < batchSize; i++) {
                    for (int z = 0; z < sizeZ; z++) {
                        for (int y = 0; y < sizeY; y++) {
                            for (int x = 0; x < sizeX; x++) {
                                double v = inputs[i][z][y][x] + bias[z][y][x];
                                double deriv = (v > 0.0) ? 1.0 : 0.0;
                                // Multiply delta by local derivative.
                                delta[i][z][y][x] *= deriv;
                                // Accumulate bias gradient.
                                biasDelta[i][z][y][x] += delta[i][z][y][x];
                            }
                        }
                    }
                }
            } else if (activationType == ActivationType.RELU5) {

                // For RELU5: output is clamped to [0,5]. Its derivative is 1 if v is in (0,5), 0 otherwise.
                for (int i = 0; i < batchSize; i++) {
                    for (int z = 0; z < sizeZ; z++) {
                        for (int y = 0; y < sizeY; y++) {
                            for (int x = 0; x < sizeX; x++) {
                                double v = inputs[i][z][y][x] + bias[z][y][x];
                                double deriv = (v > 0.0 && v < 5.0) ? 1.0 : 0.0;
                                delta[i][z][y][x] *= deriv;
                                biasDelta[i][z][y][x] += delta[i][z][y][x];
                            }
                        }
                    }
                }
            } else if (activationType == ActivationType.LEAKY_RELU5) {

                //use a leak value of 0.01 for values < 0
                double leak = 0.01;
                for (int i = 0; i < batchSize; i++) {
                    for (int z = 0; z < sizeZ; z++) {
                        for (int y = 0; y < sizeY; y++) {
                            for (int x = 0; x < sizeX; x++) {
                                double v = inputs[i][z][y][x] + bias[z][y][x];
                                double deriv;
                                if (v < 0.0) {
                                    deriv = leak;
                                } else if (v > 5.0) {
                                    deriv = 0.0;
                                } else {
                                    deriv = 1.0;
                                }
                                delta[i][z][y][x] *= deriv;
                                biasDelta[i][z][y][x] += delta[i][z][y][x];
                            }
                        }
                    }
                }

            }

            //batch normalization happens after the activation function but before dropout
            if (useBatchNormalization) {
                // TODO: implement batch normalization
                int N = batchSize * sizeZ * sizeY * sizeX;


                // Calculate gamma and beta gradients first
                for (int i = 0; i < batchSize; i++) {
                    for (int z = 0; z < sizeZ; z++) {
                        for (int y = 0; y < sizeY; y++) {
                            for (int x = 0; x < sizeX; x++) {
                                // Important: delta here is the gradient coming from the previous layer
                                betaDelta += delta[i][z][y][x];
                                gammaDelta += delta[i][z][y][x] * xHat[i][z][y][x];
                            }
                        }
                    }
                }

                // Calculate dxHat
                double[][][][] dxHat = new double[batchSize][sizeZ][sizeY][sizeX];
                for (int i = 0; i < batchSize; i++) {
                    for (int z = 0; z < sizeZ; z++) {
                        for (int y = 0; y < sizeY; y++) {
                            for (int x = 0; x < sizeX; x++) {
                                dxHat[i][z][y][x] = delta[i][z][y][x] * gamma;
                            }
                        }
                    }
                }

                // Calculate sum of dxHat and sum of (dxHat * xHat)
                double sumDxHat = 0.0;
                double sumDxHatXHat = 0.0;
                for (int i = 0; i < batchSize; i++) {
                    for (int z = 0; z < sizeZ; z++) {
                        for (int y = 0; y < sizeY; y++) {
                            for (int x = 0; x < sizeX; x++) {
                                sumDxHat += dxHat[i][z][y][x];
                                sumDxHatXHat += dxHat[i][z][y][x] * xHat[i][z][y][x];
                            }
                        }
                    }
                }

                // Store these for debugging
                mu_delta = sumDxHat / N;
                sigma2_delta = sumDxHatXHat / N;

                // Calculate final input gradients (dx)
                double invStdDev = 1.0 / Math.sqrt(sigma2_b + epsilon);

                for (int i = 0; i < batchSize; i++) {
                    for (int z = 0; z < sizeZ; z++) {
                        for (int y = 0; y < sizeY; y++) {
                            for (int x = 0; x < sizeX; x++) {
                                // Corrected formula for backprop through batch norm
                                delta[i][z][y][x] = invStdDev * (
                                        dxHat[i][z][y][x] -
                                                (sumDxHat / N) -
                                                (xHat[i][z][y][x] * sumDxHatXHat / N)
                                );
                            }
                        }
                    }
                }
            }
        }

        for (Edge inputEdge : inputEdges) {
            Log.trace(toString() + " propagating backward on input edge to " + inputEdge.inputNode.toString());
            inputEdge.propagateBackward(delta);
        }
    }

    /**
     *  This sets the node's bias to the bias parameter and then
     *  randomly initializes each incoming edge weight by using
     *  Random.nextGaussian() / sqrt(N) where N is the number
     *  of weights in the incoming edges (filters) fanning in
     *  to the layer.
     *
     *  @param initialBias is the bias to initialize this node's bias to
     */
    public void initializeWeightsAndBiasKaiming(double initialBias, int fanIn) {


        // 1) Initialize the node’s bias
        for (int z = 0; z < sizeZ; z++) {
            for (int y = 0; y < sizeY; y++) {
                for (int x = 0; x < sizeX; x++) {
                    bias[z][y][x] = initialBias;
                }
            }
        }

        // 2) Compute the range used for Kaiming/He normal initialization
        //    (some assignments simply do nextGaussian()/sqrt(fanIn), others do sqrt(2.0/fanIn)).
        double range = Math.sqrt(2.0 / fanIn);

        // 3) For each incoming edge, call the existing initializeKaiming(...) method
        for (Edge e : inputEdges) {
            if (e instanceof ConvolutionalEdge) {
                ConvolutionalEdge convEdge = (ConvolutionalEdge) e;
                convEdge.initializeKaiming(range, fanIn);
            }
        }

    }

    /**
     *  This sets the node's bias to the bias parameter and then
     *  randomly intializes each incoming edge weight uniformly
     *  at random (you can use Random.nextDouble()) between 
     *  +/- sqrt(6) / sqrt(fan_in + fan_out) 
     *
     *  Where the fan in/fan out are the number of weights in the
     *  edges (filters) fanning into/out of the layer of the node
     *
     *  @param initialBias is the bias to initialize this node's bias to
     */
    public void initializeWeightsAndBiasXavier(double initialBias, int fanIn, int fanOut) {


        // 1) Initialize the node’s bias
        for (int z = 0; z < sizeZ; z++) {
            for (int y = 0; y < sizeY; y++) {
                for (int x = 0; x < sizeX; x++) {
                    bias[z][y][x] = initialBias;
                }
            }
        }

        // 2) Compute the range for Xavier uniform initialization
        //    range = sqrt(6.0 / (fanIn + fanOut))
        double range = Math.sqrt(6.0 / (fanIn + fanOut));

        // 3) For each incoming edge, call the existing initializeXavier(...) method
        for (Edge e : inputEdges) {
            if (e instanceof ConvolutionalEdge) {
                ConvolutionalEdge convEdge = (ConvolutionalEdge) e;
                convEdge.initializeXavier(range, fanIn, fanOut);
            }
        }

    }


    /**
     * Prints concise information about this node.
     *
     * @return The node as a short string.
     */
    public String toString() {
        return "[Node - layer: " + layer + ", number: " + number + ", type: " + nodeType + ", activation: " + activationType + ", size: " + batchSize + "x" + sizeZ + "x" + sizeY + "x" + sizeX + "]";
    }
}
