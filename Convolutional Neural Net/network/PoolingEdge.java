/**
 * This class represents an PoolingEdge in a neural network. It will contain
 * the PoolingEdge's weight, and also have references to input node and output
 * nodes of this edge.
 */
package network;

import java.util.Random;

import util.Log;

public class PoolingEdge extends Edge {
    //the delta calculated by backpropagation for this edge
    public double poolDelta[][][][];

    public int batchSize;
    public int poolSize;
    public int stride;

    /**
     * This constructs a new edge in the neural network between the passed
     * parameters. It will register itself at the input and output nodes
     * through the Node.addOutgoingPoolingEdge(PoolingEdge) and Node.addIncomingPoolingEdge(PoolingEdge)
     * methods.
     *
     * @param inputNode is the input for this edge
     * @param outputNode is the output for this edge
     */
    public PoolingEdge(ConvolutionalNode inputNode, ConvolutionalNode outputNode, int poolSize, int stride) throws NeuralNetworkException {
        //we can just use the input node's batch and x y z sizes to initialize the pool delta array
        super(inputNode, outputNode, inputNode.sizeZ, inputNode.sizeY, inputNode.sizeX);
        this.inputNode = inputNode;
        this.outputNode = outputNode;
        this.batchSize = inputNode.batchSize;
        this.poolSize = poolSize;
        this.stride = stride;

        //initialize the weight and delta to 0
        poolDelta = new double[inputNode.batchSize][inputNode.sizeZ][inputNode.sizeY][inputNode.sizeX];

        if (inputNode.sizeZ != outputNode.sizeZ
               || ((inputNode.sizeY - poolSize) / stride) + 1 != (outputNode.sizeY - (2 * outputNode.padding))
               || ((inputNode.sizeX - poolSize) / stride) + 1 != (outputNode.sizeX - (2 * outputNode.padding))) {
            throw new NeuralNetworkException("Cannot connect input node " + inputNode.toString() + " to output node " + outputNode.toString() + " because sizes do not work with this pooling edge (stride: " + stride + ", pool size: " + poolSize + "), output node size should be (batchSize x" + inputNode.sizeZ + "x" + (((inputNode.sizeY - poolSize) / stride) + 1) + "x" + (((inputNode.sizeX - poolSize) / stride) + 1) + ")");
        }

    }

    /**
     * Resets the deltas for this edge
     */
    public void reset() {
        for (int i = 0; i < batchSize; i++) {
            for (int z = 0; z < sizeZ; z++) {
                for (int y = 0; y < sizeY; y++) {
                    for (int x = 0; x < sizeX; x++) {
                        poolDelta[i][z][y][x] = 0;
                    }
                }
            }
        }
    }

    /**
     * Used to get the weights of this Edge.
     * It will set the weights in the weights
     * parameter passed in starting at position, and return the number of
     * weights it set.
     *
     * @param position is the index to start setting weights in the weights parameter
     * @param weights is the array of weights we're setting.
     *
     * @return the number of weights set in the weights parameter
     */
    public int getWeights(int position, double[] weights) {
        //the pooling edge has no weights so we can just return 0
        return 0;
    }

    /**
     * Used to print gradients related to this edge, along with informationa
     * about this edge.
     * It start printing the gradients passed in starting at position, and 
     * return the number of gradients it printed.
     *
     * @param position is the index to start printing different gradients
     * @param numericGradient is the array of the numeric gradient we're printing
     * @param backpropGradient is the array of the backprop gradient we're printing
     *
     * @return the number of gradients printed by this edge
     */
    public int printGradients(int position, double[] numericGradient, double[] backpropGradient) {
        //don't print anything out, but print out this edge
        Log.info("PoolingEdge from Node [layer: " + inputNode.layer + ", number: " + inputNode.number + "] to Node [layer: " + outputNode.layer + ", number: " + outputNode.number + "] to Node - no gradients.");

        return 0;
    }


    /**
     * Used to get the deltas of this Edge. 
     * It will set the deltas in the deltas
     * parameter passed in starting at position, and return the number of
     * deltas it set.
     *
     * @param position is the index to start setting deltas in the deltas parameter
     * @param deltas is the array of deltas we're setting.
     *
     * @return the number of deltas set in the deltas parameter
     */
    public int getDeltas(int position, double[] deltas) {
        //the pooling edge has now weights, and the deltas
        //are just used to make sure the backward pass works right
        //so we don't need to return anything here either.

        return 0;
    }


    /**
     * Used to set the weights of this edge .
     * It uses the same technique as Node.getWeights
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
        return 0;
    }


    /**
     * This performs the max pooling operation by selecting the
     * maximum value from each pool of the input node's output
     * values and assigning them to the output node's input values.
     *
     * @param inputValues are the outputValues from the input node 
     * (i.e., the input values to the max pooling operation)
     */
    public void propagateForward(double[][][][] inputValues) {

        int outYNoPad = outputNode.sizeY - 2 * outputNode.padding;  // e.g. (inputNode.sizeY - poolSize)/stride + 1
        int outXNoPad = outputNode.sizeX - 2 * outputNode.padding;

        for (int i = 0; i < batchSize; i++) {
            for (int z = 0; z < sizeZ; z++) {
                // Because sizeZ is the same as inputNode.sizeZ == outputNode.sizeZ
                for (int oy = 0; oy < outYNoPad; oy++) {
                    // input start row = oy * stride
                    int inY = oy * stride;
                    for (int ox = 0; ox < outXNoPad; ox++) {
                        // input start col = ox * stride
                        int inX = ox * stride;
                        double maxVal = Double.NEGATIVE_INFINITY;

                        // Find max in the poolSize x poolSize block
                        for (int py = 0; py < poolSize; py++) {
                            for (int px = 0; px < poolSize; px++) {
                                double val = inputValues[i][z][inY + py][inX + px];
                                if (val > maxVal) {
                                    maxVal = val;
                                }
                            }
                        }

                        // Put the max in the output node's input array
                        int yOut = oy + outputNode.padding;
                        int xOut = ox + outputNode.padding;
                        outputNode.inputValues[i][z][yOut][xOut] = maxVal;
                    }
                }
            }
        }

    }


    /**
     * This takes an incoming delta from the output node
     * and propagates it backwards to the input node.
     *
     * @param delta is the delta/error from the output node.
     */
    public void propagateBackward(double[][][][] delta) {

        int outYNoPad = outputNode.sizeY - 2 * outputNode.padding;
        int outXNoPad = outputNode.sizeX - 2 * outputNode.padding;

        // Iterate over each example and channel.
        for (int i = 0; i < batchSize; i++) {
            for (int z = 0; z < sizeZ; z++) {
                // For each output pooling position.
                for (int oy = 0; oy < outYNoPad; oy++) {
                    // Starting row in the input for the pooling window.
                    int inY = oy * stride;
                    for (int ox = 0; ox < outXNoPad; ox++) {
                        // Starting column in the input for the pooling window.
                        int inX = ox * stride;
                        // Determine the maximum value and its location in the pooling window.
                        double maxVal = Double.NEGATIVE_INFINITY;
                        int maxY = -1;
                        int maxX = -1;
                        for (int py = 0; py < poolSize; py++) {
                            for (int px = 0; px < poolSize; px++) {
                                double val = inputNode.outputValues[i][z][inY + py][inX + px];
                                if (val > maxVal) {
                                    maxVal = val;
                                    maxY = inY + py;
                                    maxX = inX + px;
                                }
                            }
                        }

                        // Determine the corresponding output delta position.
                        // Recall that in propagateForward, we wrote the pooled value into:
                        //    outputNode.inputValues[i][z][oy + outputNode.padding][ox + outputNode.padding]
                        // Hence, we must offset the indices when reading delta.
                        int outYIndex = oy + outputNode.padding;
                        int outXIndex = ox + outputNode.padding;

                        // The gradient from the pooled output is routed only to the maximum location.
                        double grad = delta[i][z][outYIndex][outXIndex];
                        inputNode.delta[i][z][maxY][maxX] += grad;
                    }
                }
            }
        }
    }


}
