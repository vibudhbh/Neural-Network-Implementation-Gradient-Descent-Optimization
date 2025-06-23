/**
 * This class represents an ConvolutionalEdge in a neural network. It will contain
 * the ConvolutionalEdge's weight, and also have references to input node and output
 * nodes of this edge.
 */
package network;

import java.util.Random;

import util.Log;

public class ConvolutionalEdge extends Edge {
    //the weight for this edge
    public double weight[][][];

    //the delta calculated by backpropagation for this edge
    public double weightDelta[][][];

    /**
     * This constructs a new edge in the neural network between the passed
     * parameters. It will register itself at the input and output nodes
     * through the Node.addOutgoingConvolutionalEdge(ConvolutionalEdge) and Node.addIncomingConvolutionalEdge(ConvolutionalEdge)
     * methods.
     *
     * @param inputNode is the input for this edge
     * @param outputNode is the output for this edge
     */
    public ConvolutionalEdge(ConvolutionalNode inputNode, ConvolutionalNode outputNode, int sizeZ, int sizeY, int sizeX) throws NeuralNetworkException {
        super(inputNode, outputNode, sizeZ, sizeY, sizeX);
        this.inputNode = inputNode;
        this.outputNode = outputNode;

        if (inputNode.sizeZ - sizeZ + 1 != outputNode.sizeZ
                || inputNode.sizeY - sizeY + 1 != outputNode.sizeY - (2 * outputNode.padding)
                || inputNode.sizeX - sizeX + 1 != outputNode.sizeX - (2 * outputNode.padding)) {
            throw new NeuralNetworkException("Cannot connect input node " + inputNode.toString() + " to output node " + outputNode.toString() + " because sizes do not work with this filter (" + sizeZ + "x" + sizeY + "x" + sizeX  + "), output node size should be (batchSize x" + (inputNode.sizeZ - sizeZ + 1) + "x" + (inputNode.sizeY - sizeY + 1) + "x" + (inputNode.sizeX - sizeX + 1) + ")");
        }

        //initialize the weight and delta to 0
        weight = new double[sizeZ][sizeY][sizeX];
        weightDelta = new double[sizeZ][sizeY][sizeX];
    }

    /**
     * Resets the deltas for this edge
     */
    public void reset() {
        //Log.info("resetting convolutional edge with sizeZ: " + sizeZ + ", sizeY: " + sizeY + ", sizeX: " + sizeX);
        for (int z = 0; z < sizeZ; z++) {
            for (int y = 0; y < sizeY; y++) {
                for (int x = 0; x < sizeX; x++) {
                    weightDelta[z][y][x] = 0;
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
        int weightCount = 0;

        for (int z = 0; z < sizeZ; z++) {
            for (int y = 0; y < sizeY; y++) {
                for (int x = 0; x < sizeX; x++) {
                    weights[position + weightCount] = weight[z][y][x];
                    weightCount++;
                }
            }
        }

        return weightCount;
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
        Log.info("ConvolutionalEdge from Node [layer: " + inputNode.layer + ", number: " + inputNode.number + "] to Node [layer: " + outputNode.layer + ", number: " + outputNode.number + "] to Node:");

        int count = 0;

        for (int z = 0; z < sizeZ; z++) {
            for (int y = 0; y < sizeY; y++) {
                for (int x = 0; x < sizeX; x++) {
                    Log.info("\tweights[" + z + "][" + y + "][" + x + "]: "+ Log.twoGradients(numericGradient[position + count], backpropGradient[position + count]));
                    count++;
                }
            }
        }

        return count;
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
        int deltaCount = 0;

        for (int z = 0; z < sizeZ; z++) {
            for (int y = 0; y < sizeY; y++) {
                for (int x = 0; x < sizeX; x++) {
                    deltas[position + deltaCount] = weightDelta[z][y][x];
                    deltaCount++;
                }
            }
        }

        return deltaCount;
    }


    /**
     * Used to set the weights of this Edge.
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
        int weightCount = 0;

        for (int z = 0; z < sizeZ; z++) {
            for (int y = 0; y < sizeY; y++) {
                for (int x = 0; x < sizeX; x++) {
                    weight[z][y][x] = weights[position + weightCount];
                    weightCount++;
                }
            }
        }

        return weightCount;
    }



    /**
     * This initializes the weights of this ConvolutionalEdge (Filter) by
     * the range calculated by it's output node (which should be sqrt(2)/sqrt(all incoming edge filter sizes).
     *
     * @param range is sqrt(2)/sqrt(sum of output node incoming filter sizes)
     */
    public void initializeKaiming(double range, int fanIn) {


        // Kaiming normal: typically nextGaussian() * sqrt(2.0/fanIn).
        // In this design, "range" is already passed as sqrt(2.0/fanIn).
        // Kaiming initialization - Normally distributed with stddev = sqrt(2/fanIn)
        for (int z = 0; z < sizeZ; z++) {
            for (int y = 0; y < sizeY; y++) {
                for (int x = 0; x < sizeX; x++) {
                    // Use Gaussian distribution scaled by range (which should be sqrt(2/fanIn))
                    weight[z][y][x] = inputNode.generator.nextGaussian() * range;

                    // Safety check - limit initial weights to prevent extreme values
                    if (weight[z][y][x] > 1.0) weight[z][y][x] = 1.0;
                    if (weight[z][y][x] < -1.0) weight[z][y][x] = -1.0;
                }
            }
        }

    }

    /**
     * This initializes the weights of this ConvolutionalEdge (Filter) by
     * uniformly within the range calculated by it's output node (which 
     * should be between negative and positive sqrt(6)/sqrt(all incoming 
     * and outgoing edge filter sizes).
     *
     * @param range is sqrt(6)/sqrt(sum of output node incoming and outgoing filter sizes)
     */
    public void initializeXavier(double range, int fanIn, int fanOut) {

        // Xavier uniform: [-range, range], where range = sqrt(6/(fanIn+fanOut))
        for (int z = 0; z < sizeZ; z++) {
            for (int y = 0; y < sizeY; y++) {
                for (int x = 0; x < sizeX; x++) {
                    // Generate uniform random value in [-range, range]
                    double r = (inputNode.generator.nextDouble() * 2.0 * range) - range;

                    // Safety check - limit initial weights further to prevent extreme values
                    if (r > 0.5) r = 0.5;
                    if (r < -0.5) r = -0.5;

                    weight[z][y][x] = r;
                }
            }
        }

    }


    /**
     * This takes an incoming delta from the output node
     * and propagates it backwards to the input node.
     *
     * @param inputValues is the delta/error from the output node.
     */
    public void propagateForward(double[][][][] inputValues) {


        int outZ = outputNode.sizeZ;                       // e.g. inputNode.sizeZ - sizeZ + 1
        int outYNoPad = outputNode.sizeY - 2 * outputNode.padding;
        int outXNoPad = outputNode.sizeX - 2 * outputNode.padding;
        // outYNoPad should be (inputNode.sizeY - sizeY + 1), etc.

        for (int i = 0; i < inputNode.batchSize; i++) {
            for (int oz = 0; oz < outZ; oz++) {
                for (int oy = 0; oy < outYNoPad; oy++) {
                    for (int ox = 0; ox < outXNoPad; ox++) {

                        double sum = 0.0;
                        // sum over the filter
                        for (int fz = 0; fz < sizeZ; fz++) {
                            for (int fy = 0; fy < sizeY; fy++) {
                                for (int fx = 0; fx < sizeX; fx++) {
                                    double inVal = inputValues[i][oz + fz][oy + fy][ox + fx];
                                    double w = weight[fz][fy][fx];
                                    sum += inVal * w;
                                }
                            }
                        }

                        // Now place that sum into the output node’s input array
                        // account for padding on the output node
                        int yOut = oy + outputNode.padding;
                        int xOut = ox + outputNode.padding;
                        outputNode.inputValues[i][oz][yOut][xOut] += sum;
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

        int batch = inputNode.batchSize;
        int outZ = outputNode.sizeZ;
        int outYNoPad = outputNode.sizeY - 2 * outputNode.padding;
        int outXNoPad = outputNode.sizeX - 2 * outputNode.padding;

        // 1. Accumulate weight gradients.
        for (int i = 0; i < batch; i++) {
            for (int oz = 0; oz < outZ; oz++) {
                for (int oy = 0; oy < outYNoPad; oy++) {
                    for (int ox = 0; ox < outXNoPad; ox++) {
                        // Account for padding in the output node
                        double d = delta[i][oz][oy + outputNode.padding][ox + outputNode.padding];
                        for (int fz = 0; fz < sizeZ; fz++) {
                            for (int fy = 0; fy < sizeY; fy++) {
                                for (int fx = 0; fx < sizeX; fx++) {
                                    double inVal = inputNode.outputValues[i][oz + fz][oy + fy][ox + fx];
                                    weightDelta[fz][fy][fx] += inVal * d;
                                }
                            }
                        }
                    }
                }
            }
        }

        // 2. Propagate delta back to the input node.
        for (int i = 0; i < batch; i++) {
            for (int oz = 0; oz < outZ; oz++) {
                for (int oy = 0; oy < outYNoPad; oy++) {
                    for (int ox = 0; ox < outXNoPad; ox++) {
                        // Account for padding in the output node
                        double d = delta[i][oz][oy + outputNode.padding][ox + outputNode.padding];
                        for (int fz = 0; fz < sizeZ; fz++) {
                            for (int fy = 0; fy < sizeY; fy++) {
                                for (int fx = 0; fx < sizeX; fx++) {
                                    int inZ = oz + fz;
                                    int inY = oy + fy;
                                    int inX = ox + fx;
                                    inputNode.delta[i][inZ][inY][inX] += weight[fz][fy][fx] * d;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

}
