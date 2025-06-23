package network;

import java.util.ArrayList;
import java.util.List;

import data.Sequence;
import data.CharacterSequence;
import data.TimeSeries;

import util.Log;


public class RecurrentNeuralNetwork {
    //this is the loss function for the output of the neural network
    LossFunction lossFunction;
    
    //this is the maximum length of any sequence
    int maxSequenceLength;

    //this is the total number of weights in the neural network
    int numberWeights;
    
    //layers contains all the nodes in the neural network
    RecurrentNode[][] layers;

    public RecurrentNeuralNetwork(int inputLayerSize, int[] hiddenLayerSizes, int outputLayerSize, int maxSequenceLength, RNNNodeType rnnNodeType, LossFunction lossFunction) {
        this.lossFunction = lossFunction;
        this.maxSequenceLength = maxSequenceLength;

        //the number of layers in the neural network is 2 plus the number of hidden layers,
        //one additional for the input, and one additional for the output.

        //create the outer array of the 2-dimensional array of nodes
        layers = new RecurrentNode[hiddenLayerSizes.length + 2][];
        
        //we will progressively calculate the number of weights as we create the network. the
        //number of edges will be equal to the number of hidden nodes (each has a bias weight, but
        //the input and output nodes do not) plus the number of edges
        numberWeights = 0;

        Log.info("creating a neural network with " + hiddenLayerSizes.length + " hidden layers, for max sequence length: " + maxSequenceLength + ".");
        for (int layer = 0; layer < layers.length; layer++) {
            
            //determine the layer size depending on the layer number, 0 is the
            //input layer, and the last layer is the output layer, all others
            //are hidden layers
            int layerSize;
            NodeType nodeType;
            if (layer == 0) {
                //this is the input layer
                layerSize = inputLayerSize;
                nodeType = NodeType.INPUT;
                Log.info("input layer " + layer + " has " + layerSize + " nodes.");

            } else if (layer < layers.length - 1) {
                //this is a hidden layer
                layerSize = hiddenLayerSizes[layer - 1];
                nodeType = NodeType.HIDDEN;
                Log.info("hidden layer " + layer + " has " + layerSize + " nodes.");

            } else {
                //this is the output layer
                layerSize = outputLayerSize;
                nodeType = NodeType.OUTPUT;
                Log.info("output layer " + layer + " has " + layerSize + " nodes.");
            }

            //create the layer with the right length and right node types
            layers[layer] = new RecurrentNode[layerSize];

            for (int j = 0; j < layers[layer].length; j++) {
                if (nodeType == NodeType.INPUT) {
                    //input nodes do not have an activation function applied to them
                    layers[layer][j] = new RecurrentNode(layer, j /*i is the node number*/, nodeType, maxSequenceLength, ActivationType.LINEAR);

                } else {
                    switch (rnnNodeType) {
                        case LINEAR:
                            layers[layer][j] = new RecurrentNode(layer, j /*i is the node number*/, nodeType, maxSequenceLength, ActivationType.LINEAR);
                            //increment the number of weights here because some node types will have a different amount
                            //linear nodes don't have bias in the output layer
                            if (nodeType == NodeType.HIDDEN) numberWeights++;
                            break;
                        case SIGMOID:
                            layers[layer][j] = new RecurrentNode(layer, j /*i is the node number*/, nodeType, maxSequenceLength, ActivationType.SIGMOID);
                            //increment the number of weights here because some node types will have a different amount
                            //sigmoid nodes don't have bias in the output layer
                            if (nodeType == NodeType.HIDDEN) numberWeights++;
                            break;
                        case TANH:
                            layers[layer][j] = new RecurrentNode(layer, j /*i is the node number*/, nodeType, maxSequenceLength, ActivationType.TANH);
                            //increment the number of weights here because some node types will have a different amount
                            //tanh nodes don't have bias in the output layer
                            if (nodeType == NodeType.HIDDEN) numberWeights++;
                            break;
                        case LSTM:
                            //TODO: implement this for Programming Assignment 2 - Part 4
                            //increment the number of weights here because some node types will have a different amount
                            layers[layer][j] = new LSTMNode(layer, j /*i is the node number*/, nodeType, maxSequenceLength);
                            numberWeights += 11;
                            break;
                        case GRU:
                            //TODO: BONUS implement this for Programming Assignment 2 - Part 4
                            break;
                        case MGU:
                            //TODO: BONUS implement this for Programming Assignment 2 - Part 4
                            break;
                        case UGRNN:
                            //TODO: BONUS implement this for Programming Assignment 2 - Part 4
                            break;
                        case DELTA:
                            //TODO: BONUS implement this for Programming Assignment 2 - Part 4
                            break;
                        default:
                            Log.fatal("Trying to create RNN with unknown RNNNodeType - this should never happen!");
                            System.exit(1);
                    }
                }
            }
        }
    }

    /**
     * This gets the number of weights in the RecurrentNeuralNetwork, which should
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
     * Gets the name for each weight so things can be debugged easier
     * @return a array of strings, each corresponding to what edge/bias each weight represents from the
     *      results of the getWeights() method.
     */
    public String[] getWeightNames() throws NeuralNetworkException {
        String[] weightNames = new String[numberWeights];

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
                int nWeights = layers[layer][nodeNumber].getWeightNames(position, weightNames);
                position += nWeights;

                if (position > numberWeights) {
                    throw new NeuralNetworkException("The numberWeights field of the RecurrentNeuralNetwork was (" + numberWeights + ") but when getting the weight names there were more hidden nodes and edges than numberWeights. This should not happen unless numberWeights is not being updated correctly.");
                }
            }
        }

        return weightNames;
    }


    /**
     * This returns an array of every weight (including biases) in the RecurrentNeuralNetwork.
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
                    throw new NeuralNetworkException("The numberWeights field of the RecurrentNeuralNetwork was (" + numberWeights + ") but when getting the weights there were more hidden nodes and edges than numberWeights. This should not happen unless numberWeights is not being updated correctly.");
                }
            }
        }

        return weights;
    }

    /**
     * This sets every weight (including biases) in the RecurrentNeuralNetwork, it sets them in
     * the same order that they are retreived by the getWeights method.
     * This will be very useful in backpropagation and sanity checking. 
     *
     * @throws NeuralNetworkException if numberWeights was not calculated correctly. This shouldn't happen.
     */
    public void setWeights(double[] newWeights) throws NeuralNetworkException {
        if (numberWeights != newWeights.length) {
            throw new NeuralNetworkException("Could not setWeights because the number of new weights: " + newWeights.length + " was not equal to the number of weights in the RecurrentNeuralNetwork: " + numberWeights);
        }

        int position = 0;
        for (int layer = 0; layer < layers.length; layer++) {
            for (int nodeNumber = 0; nodeNumber < layers[layer].length; nodeNumber++) {
                int nWeights = layers[layer][nodeNumber].setWeights(position, newWeights);
                position += nWeights;

                if (position > numberWeights) {
                    throw new NeuralNetworkException("The numberWeights field of the RecurrentNeuralNetwork was (" + numberWeights + ") but when setting the weights there were more hidden nodes and edges than numberWeights. This should not happen unless numberWeights is not being updated correctly.");
                }
            }
        }
    }

    /**
     * This returns an array of every weight (including biases) in the RecurrentNeuralNetwork.
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
                    throw new NeuralNetworkException("The numberWeights field of the RecurrentNeuralNetwork was (" + numberWeights + ") but when getting the deltas there were more hidden nodes and edges than numberWeights. This should not happen unless numberWeights is not being updated correctly.");
                }
            }
        }

        return deltas;
    }


    /**
     * This adds edges to the RecurrentNeuralNetwork, connecting each node
     * in a layer to each node in the subsequent layer
     */
    public void connectFully() throws NeuralNetworkException {
        //create outgoing edges from the input layer to the last hidden layer,
        //the output layer will not have outgoing edges
        for (int layer = 0; layer < layers.length - 1; layer++) {

            //iterate over the nodes in the current layer
            for (int inputNodeNumber = 0; inputNodeNumber < layers[layer].length; inputNodeNumber++) {

                //iterate over the nodes in the next layer
                for (int outputNodeNumber = 0; outputNodeNumber < layers[layer + 1].length; outputNodeNumber++) {
                    RecurrentNode inputNode = layers[layer][inputNodeNumber];
                    RecurrentNode outputNode = layers[layer + 1][outputNodeNumber];
                    new Edge(inputNode, outputNode);

                    //as we added an edge, the number of weights should increase by 1
                    numberWeights++;
                    Log.trace("numberWeights now: " + numberWeights);
                }
            }
        }
    }

    /**
     * Makes this RNN a Jordan network by creating a RecurrentEdge between
     * every output node and every hidden node.
     *
     * @param timeSkip is how many time steps to skip for the recurrent connection
     */
    public void connectJordan(int timeSkip) throws NeuralNetworkException {
        //You need to implement this for Programming Assignment 2 - Part 1

        // 1) Gather all output nodes
        List<RecurrentNode> outputNodes = new ArrayList<>();
        for (int layerIndex = 0; layerIndex < layers.length; layerIndex++) {
            for (int nodeIndex = 0; nodeIndex < layers[layerIndex].length; nodeIndex++) {
                if (layers[layerIndex][nodeIndex].nodeType == NodeType.OUTPUT) {
                    outputNodes.add(layers[layerIndex][nodeIndex]);
                }
            }
        }

        // 2) Gather all hidden nodes
        List<RecurrentNode> hiddenNodes = new ArrayList<>();
        for (int layerIndex = 0; layerIndex < layers.length; layerIndex++) {
            for (int nodeIndex = 0; nodeIndex < layers[layerIndex].length; nodeIndex++) {
                if (layers[layerIndex][nodeIndex].nodeType == NodeType.HIDDEN) {
                    hiddenNodes.add(layers[layerIndex][nodeIndex]);
                }
            }
        }

        // 3) Create a RecurrentEdge from each output node to each hidden node
        for (RecurrentNode outNode : outputNodes) {
            for (RecurrentNode hidNode : hiddenNodes) {
                RecurrentEdge re = new RecurrentEdge(outNode, hidNode, timeSkip);
                // 4) Increment your total weight counter (or store the new edge, etc.)
                numberWeights++;
            }
        }

        Log.trace("Added Jordan recurrent edges from all output nodes to all hidden nodes.");
    }

    /**
     * Makes this RNN an Elman network by creating a RecurrentEdge from 
     * every hidden node in a layer to every other hidden node in that layer
     *
     * @param timeSkip is how many time steps to skip for the recurrent connection
     */
    public void connectElman(int timeSkip) throws NeuralNetworkException {
        //You need to implement this for Programming Assignment 2 - Part 1

        // For each layer in your network
        for (int layerIndex = 0; layerIndex < layers.length; layerIndex++) {
            // 1) Gather all hidden nodes in this layer
            List<RecurrentNode> hiddenNodesInLayer = new ArrayList<>();
            for (int nodeIndex = 0; nodeIndex < layers[layerIndex].length; nodeIndex++) {
                RecurrentNode node = layers[layerIndex][nodeIndex];
                if (node.nodeType == NodeType.HIDDEN) {
                    hiddenNodesInLayer.add(node);
                }
            }

            // 2) Create a RecurrentEdge between every pair of hidden nodes in this layer
            //    (including self-edges, i.e., from a node to itself)
            for (RecurrentNode sourceHidden : hiddenNodesInLayer) {
                for (RecurrentNode targetHidden : hiddenNodesInLayer) {
                    new RecurrentEdge(sourceHidden, targetHidden, timeSkip);
                    numberWeights++;  // track or increment your total weight count
                }
            }
        }
        Log.trace("Added Elman recurrent edges among hidden nodes in each layer.");
    }


    /**
     * This will create an Edge between the node with number inputNumber on the inputLayer to the
     * node with the outputNumber on the outputLayer.
     *
     * @param inputLayer the layer of the input node
     * @param inputNumber the number of the input node on layer inputLayer
     * @param outputLayer the layer of the output node
     * @param outputNumber the number of the output node on layer outputLayer
     */
    public void connectNodes(int inputLayer, int inputNumber, int outputLayer, int outputNumber) throws NeuralNetworkException {
        if (inputLayer >= outputLayer) {
            throw new NeuralNetworkException("Cannot create an Edge between input layer " + inputLayer + " and output layer " + outputLayer + " because the layer of the input node must be less than the layer of the output node.");
        //} else if (outputLayer != inputLayer + 1) {
            //throw new NeuralNetworkException("Cannot create an Edge between input layer " + inputLayer + " and output layer " + outputLayer + " because the layer of the output node must be the next layer in the network.");
        }
        // Validating that the layers are within bounds
        if (inputLayer < 0 || inputLayer >= layers.length) {
            throw new NeuralNetworkException("Invalid inputLayer index: " + inputLayer);
        }
        // Validating that the node indices are within bounds
        if (inputNumber < 0 || inputNumber >= layers[inputLayer].length) {
            throw new NeuralNetworkException("Invalid inputNumber index: " + inputNumber);
        }
        if (outputNumber < 0 || outputNumber >= layers[outputLayer].length) {
            throw new NeuralNetworkException("Invalid outputNumber index: " + outputNumber);
        }

        // Creating a new edge from the specified input node to the specified output node
        RecurrentNode inputNode = layers[inputLayer][inputNumber];
        RecurrentNode outputNode = layers[outputLayer][outputNumber];
        new Edge(inputNode, outputNode);

        // Updating the number of weights in the neural network
        numberWeights++;
        Log.trace("Connected node (" + inputLayer + "," + inputNumber + ") to "
                + "(" + outputLayer + "," + outputNumber + "). numberWeights now: " + numberWeights);
        //Complete this function. BONUS: allow it to it create edges that can skip layers
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
        //You need to implement this for Programming Assignment 2 - Part 2

        // Go through each layer and each node
        for (int layerIndex = 0; layerIndex < layers.length; layerIndex++) {
            for (int nodeIndex = 0; nodeIndex < layers[layerIndex].length; nodeIndex++) {
                RecurrentNode node = layers[layerIndex][nodeIndex];

                // Typically, only hidden nodes have biases in your code (based on comments).
                // If your assignment wants to initialize output-node biases as well, adapt accordingly.
                if (node.nodeType == NodeType.HIDDEN) {
                    // 1) Compute fanIn, fanOut for the node.
                    //    fanIn = # of inputEdges + # of inputRecurrentEdges
                    //    fanOut = # of outputEdges + # of outputRecurrentEdges
                    int fanIn = node.inputEdges.size() + node.inputRecurrentEdges.size();
                    int fanOut = node.outputEdges.size() + node.outputRecurrentEdges.size();

                    // 2) Depending on "type", call the appropriate init method.
                    if (type.equalsIgnoreCase("kaiming")) {
                        node.initializeWeightsAndBiasKaiming(fanIn, bias);
                    } else if (type.equalsIgnoreCase("xavier")) {
                        // We need both fanIn and fanOut
                        node.initializeWeightsAndBiasXavier(fanIn, fanOut, bias);
                    } else {
                        // If your assignment doesn't require handling other strings, you can throw an exception
                        Log.error("Unknown initialization type: " + type);
                    }
                }
            }
        }
    }



    /**
     * This performs a forward pass through the neural network given
     * inputs from the input instance.
     *
     * @param sequence is the data set instance to pass through the network
     *
     * @return the sum of the output of all output nodes
     */
    public double forwardPass(Sequence sequence) throws NeuralNetworkException {
        //be sure to reset before doing a forward pass
        reset();

        //set input values differently for time series and character sequences
        if (sequence instanceof CharacterSequence) {
            CharacterSequence characterSequence = (CharacterSequence)sequence;

            //set the input nodes for each time step in the CharacterSequence
            for (int timeStep = 0; timeStep < characterSequence.getLength() - 1; timeStep++) {
                int value = characterSequence.valueAt(timeStep);
                for (int number = 0; number < layers[0].length; number++) {
                    RecurrentNode inputNode = layers[0][number];
                    if (value == number) {
                        //the value will be 0..n where n is the number of input nodes (and possible character values)
                        inputNode.postActivationValue[timeStep] = 1.0;
                    } else {
                        inputNode.postActivationValue[timeStep] = 0.0;
                    }
                }
            }

        } else if (sequence instanceof TimeSeries) {
            TimeSeries series = (TimeSeries)sequence;

            //set the input nodes for each time step in the TimeSeries
            for (int timeStep = 0; timeStep < series.getLength() - 1; timeStep++) {
                for (int number = 0; number < layers[0].length; number++) {
                    RecurrentNode inputNode = layers[0][number];
                    inputNode.postActivationValue[timeStep] = series.getInputValue(timeStep, number);
                }
            }

        }

        //You need to implement propagating forward for each node (output nodes need
        //to be propagated forward for their recurrent connections to further time steps)
        //for Programming Assignment 2 - Part 1
        //NOTE: This shouldn't need to be changed for Programming Assignment 2 - Parts 2 or 3
        // The total sequence length you want to iterate over
        int sequenceLength;
        if (sequence instanceof CharacterSequence) {
            sequenceLength = ((CharacterSequence)sequence).getLength();
        } else if (sequence instanceof TimeSeries) {
            sequenceLength = ((TimeSeries)sequence).getLength();
        } else {
            throw new NeuralNetworkException("Unknown sequence type: " + sequence.getClass().getSimpleName());
        }

        // For each time step (excluding the final one if your task is next-step prediction),
        // propagate forward through every layer and node
        for (int t = 0; t < sequenceLength - 1; t++) {
            // Layer 0 is typically the input layer; layer (layers.length-1) is the output layer
            for (int layerIndex = 0; layerIndex < layers.length; layerIndex++) {
                // For each node in this layer
                for (int nodeIndex = 0; nodeIndex < layers[layerIndex].length; nodeIndex++) {
                    RecurrentNode node = layers[layerIndex][nodeIndex];
                    // Let the node handle how it accumulates preActivation, adds bias, applies activation,
                    // and propagates forward (both feedforward and recurrent edges).
                    node.propagateForward(t);
                }
            }
        }

        //The following is needed for Programming Assignment 2 - Part 1
        int outputLayer = layers.length - 1;
        int nOutputs = layers[outputLayer].length;

        //note that the target value for any time step is the sequence value at that time step + 1
        //this means you should only go up to length - 1 time steps in calculating the loss
        double lossSum = 0;
        if (sequence instanceof CharacterSequence) {
            //calculate the loss functions for character sequences
            CharacterSequence characterSequence = (CharacterSequence)sequence;

            if (lossFunction == LossFunction.NONE) {
                //for no loss function we can just return the sum of the outputs overall time steps, and can set each output node's delta to 1
                for (int timeStep = 0; timeStep < characterSequence.getLength() - 1; timeStep++) {
                    for (int number = 0; number < nOutputs; number++) {
                        RecurrentNode outputNode = layers[outputLayer][number];
                        lossSum += outputNode.postActivationValue[timeStep];
                        outputNode.delta[timeStep] = 1.0;
                    }
                }

            } else if (lossFunction == LossFunction.SVM) {
                //Implement this for Programming Assignment 2 - Part 1
                for (int timeStep = 0; timeStep < characterSequence.getLength() - 1; timeStep++) {
                    int correctLabel = characterSequence.valueAt(timeStep + 1);

                    // We'll read the output node "scores" from postActivationValue[t].
                    double correctScore = layers[outputLayer][correctLabel].postActivationValue[timeStep];

                    // Track how many classes violate the margin
                    int violationsCount = 0;

                    // First, clear all deltas for this time step
                    for (int i = 0; i < nOutputs; i++) {
                        layers[outputLayer][i].delta[timeStep] = 0.0;
                    }

                    // Now compute hinge loss for each class
                    for (int j = 0; j < nOutputs; j++) {
                        if (j == correctLabel) continue;

                        double score_j = layers[outputLayer][j].postActivationValue[timeStep];
                        double margin = 1.0 + score_j - correctScore;
                        if (margin > 0) {
                            // Contributes to the loss
                            lossSum += margin;

                            // The gradient w.r.t. node j’s score is +1
                            layers[outputLayer][j].delta[timeStep] += 1.0;
                            // The gradient w.r.t. the correct class c’s score is -1
                            violationsCount++;
                        }
                    }

                    // Now apply the cumulative effect on the correct label's delta
                    layers[outputLayer][correctLabel].delta[timeStep] -= violationsCount;
                }
            } else if (lossFunction == LossFunction.SOFTMAX) {
                //Implement this for Programming Assignment 2 - Part 1
                for (int timeStep = 0; timeStep < characterSequence.getLength() - 1; timeStep++) {
                    // 1) Gather raw outputs (logits) from each output node.
                    //    In your code, you might be storing them in postActivationValue[t].
                    //    If your activationType for output nodes was LINEAR, that’s fine: they’re just raw scores.
                    double[] logits = new double[nOutputs];
                    for (int i = 0; i < nOutputs; i++) {
                        logits[i] = layers[outputLayer][i].postActivationValue[timeStep];
                    }

                    // 2) Compute softmax over these logits
                    double maxLogit = Double.NEGATIVE_INFINITY;
                    for (double val : logits) {
                        if (val > maxLogit) maxLogit = val;
                    }
                    // Subtract max for numerical stability
                    double sumExp = 0;
                    for (int i = 0; i < nOutputs; i++) {
                        logits[i] = Math.exp(logits[i] - maxLogit);
                        sumExp += logits[i];
                    }
                    // Now logits[i] holds exp(score[i]), sumExp is the denominator

                    // 3) Identify the correct label (character) at timeStep+1
                    int correctLabel = characterSequence.valueAt(timeStep + 1);

                    // 4) Cross-entropy loss = -log(prob_of_correct_label).
                    //    prob_of_correct_label = logits[correctLabel] / sumExp
                    double probCorrect = logits[correctLabel] / sumExp;
                    lossSum += -Math.log(probCorrect);

                    // 5) Compute deltas (gradients) for each output node at timeStep
                    //    delta[i] = p_i - 1 if i == correctLabel, else p_i
                    for (int i = 0; i < nOutputs; i++) {
                        double p_i = logits[i] / sumExp;
                        double deltaVal = p_i;
                        if (i == correctLabel) {
                            deltaVal = p_i - 1.0; // derivative of -log(p_correct)
                        }
                        layers[outputLayer][i].delta[timeStep] = deltaVal;
                    }
                }
            } else {
                throw new NeuralNetworkException("Could not do a CharacterSequence forward pass on RecurrentNeuralNetwork because lossFunction was unknown or invalid: " + lossFunction);
            }
        } else if (sequence instanceof TimeSeries) {
            TimeSeries series = (TimeSeries)sequence;

            if (lossFunction == LossFunction.NONE) {
                //for no loss function we can just return the sum of the outputs overall time steps, and can set each output node's delta to 1
                for (int timeStep = 0; timeStep < series.getLength() - 1; timeStep++) {
                    for (int number = 0; number < nOutputs; number++) {
                        RecurrentNode outputNode = layers[outputLayer][number];
                        lossSum += outputNode.postActivationValue[timeStep];
                        outputNode.delta[timeStep] = 1.0;
                    }
                }

            } else if (lossFunction == LossFunction.L1_NORM) {
                //Implement this for Programming Assignment 2 - Part 3

                // For regression L1 norm loss, we sum the absolute error over every time step.
                for (int timeStep = 0; timeStep < series.getLength()-1; timeStep++) {
                    for (int j = 0; j < nOutputs; j++) {
                        RecurrentNode outputNode = layers[outputLayer][j];
                        double prediction = outputNode.postActivationValue[timeStep];
                        double target = series.getOutputValue(timeStep+1, j);
                        double diff = prediction - target;
                        lossSum += Math.abs(diff);
                        // Set subgradient: sign(diff)
                        if (diff > 0) {
                            outputNode.delta[timeStep] = 1.0;
                        } else if (diff < 0) {
                            outputNode.delta[timeStep] = -1.0;
                        } else {
                            outputNode.delta[timeStep] = 0.0;
                        }
                    }
                }
            } else if (lossFunction == LossFunction.L2_NORM) {
                double totalL2 = 0.0;
                int T = series.getLength();

                // For next-step prediction, go up to T-1 and compare output(t) to target(t+1)
                for (int t = 0; t < T - 1; t++) {
                    // 1) Sum squared errors for this time step
                    double sumSquares = 0.0;
                    for (int j = 0; j < nOutputs; j++) {
                        RecurrentNode outputNode = layers[outputLayer][j];
                        double predicted = outputNode.postActivationValue[t];
                        double target = series.getOutputValue(t + 1, j);
                        double diff = predicted - target;
                        sumSquares += diff * diff;
                    }

                    // 2) local L2 norm at time t
                    double l2_t = Math.sqrt(sumSquares);

                    // 3) Add to total L2
                    totalL2 += l2_t;

                    // 4) If l2_t > 1e-12, set deltas
                    if (l2_t > 1e-12) {
                        for (int j = 0; j < nOutputs; j++) {
                            RecurrentNode outputNode = layers[outputLayer][j];
                            double predicted = outputNode.postActivationValue[t];
                            double target = series.getOutputValue(t + 1, j);
                            double diff = predicted - target;
                            // derivative wrt z_j(t) is diff / l2_t
                            outputNode.delta[t] = diff / l2_t;
                        }
                    } else {
                        // If all predictions match targets exactly at this step, delta = 0
                        for (int j = 0; j < nOutputs; j++) {
                            layers[outputLayer][j].delta[t] = 0.0;
                        }
                    }
                }

                // The final L2 loss is the sum of each local L2
                lossSum = totalL2;

            } else {
                throw new NeuralNetworkException("Could not do a TimeSeries forward pass on RecurrentNeuralNetwork because lossFunction was unknown: " + lossFunction);
            }
        }

        return lossSum;
    }

    /**
     * This performs multiple forward passes through the neural network
     * by multiple instances are returns the output sum.
     *
     * @param sequences is the set of CharacterSequences to pass through the network
     *
     * @return the sum of their outputs
     */
    public double forwardPass(List<Sequence> sequences) throws NeuralNetworkException {
        double sum = 0.0;

        for (Sequence sequence : sequences) {
            sum += forwardPass(sequence);
        }

        return sum;
    }

    /**
     * This performs multiple forward passes through the recurrent neural network
     * and calculates how many of the character predictions for every
     * sequence were classified correctly.
     *
     * @param sequences is the set of CharacterSequences to pass through the network
     *
     * @return a percentage (between 0 and 1) of how many character predictions were
     * correctly classified
     */
    public double calculateAccuracy(List<CharacterSequence> sequences) throws NeuralNetworkException {
        // need to implement this for Programming Assignment 2 - Part 2
        //the output node with the maximum value is the predicted class
        //you need to sum up how many of these match the actual class
        //for each time step of each sequence, and then calculate: 
        //num correct / total
        //to get a percentage accuracy
        int numCorrect = 0;
        int numTotal = 0;

        int outputLayerIndex = layers.length - 1;
        int nOutputs = layers[outputLayerIndex].length;

        // For each sequence in the dataset
        for (CharacterSequence seq : sequences) {
            // 1) Forward pass
            forwardPass(seq);

            // 2) For each time step we predicted
            for (int t = 0; t < seq.getLength() - 1; t++) {
                // 3) Find the argmax among the output nodes
                int predicted = -1;
                double bestScore = Double.NEGATIVE_INFINITY;
                for (int outIdx = 0; outIdx < nOutputs; outIdx++) {
                    double score = layers[outputLayerIndex][outIdx].postActivationValue[t];
                    if (score > bestScore) {
                        bestScore = score;
                        predicted = outIdx;
                    }
                }

                // 4) The true label is the next character in the sequence
                int trueLabel = seq.valueAt(t + 1);

                if (predicted == trueLabel) {
                    numCorrect++;
                }
                numTotal++;
            }
        }

        // Return accuracy as a fraction in [0,1]
        if (numTotal == 0) {
            // Avoid division by zero if sequences are empty
            return 0.0;
        }
        return (double) numCorrect / (double) numTotal;
    }


    /**
     * This gets the output values of the neural network 
     * after a forward pass, this will be a 2 dimensional array for
     * an RNN because we'll have outputs for each time step
     *
     * @param sequence is the CharacterSequence which generated the the output values, we need this to get the length of the sequence
     *
     * @return a two dimensional array of the output values from this neural network for each time step
     */
    public double[][] getOutputValues(Sequence sequence) {
        //the number of output values is the number of output nodes
        int outputLayer = layers.length - 1;
        int nOutputs = layers[outputLayer].length;

        double[][] outputValues = new double[sequence.getLength() - 1][nOutputs];

        //we will have 1 less output because we're predicting one character ahead
        for (int timeStep = 0; timeStep < sequence.getLength() - 1; timeStep++) {
            for (int number = 0; number < nOutputs; number++) {
                outputValues[timeStep][number] = layers[outputLayer][number].postActivationValue[timeStep];
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
     */
    public double[] getNumericGradient(Sequence sequence) throws NeuralNetworkException {
        //You need to implement this for Programming Assignment 2 - Part 2

        // 1) Get the current weights
        double[] originalWeights = getWeights();
        double[] numericGrad = new double[originalWeights.length];

        // 2) For each weight
        for (int i = 0; i < originalWeights.length; i++) {
            // +h
            double oldWeight = originalWeights[i];

            // w_i + H
            originalWeights[i] = oldWeight + H;
            setWeights(originalWeights);
            double fPlus = forwardPass(sequence);

            // w_i - H
            originalWeights[i] = oldWeight - H;
            setWeights(originalWeights);
            double fMinus = forwardPass(sequence);

            // numeric gradient
            double grad_i = (fPlus - fMinus) / (2.0 * H);
            numericGrad[i] = grad_i;

            // Restore the original weight
            originalWeights[i] = oldWeight;
            setWeights(originalWeights);
        }

        return numericGrad;
    }

    /**
     * This calculates the gradient of the neural network with it's current
     * weights for a given DataSet Instance using the finite difference method:
     * gradient[i] = (f(x where x[i] = x[i] + H) - f(x where x[i] = x[i] - H)) / 2h
     */
    public double[] getNumericGradient(List<Sequence> sequences) throws NeuralNetworkException {
        //You need to implement this for Programming Assignment 2 - Part 2

        // 1) Get the current weights
        double[] originalWeights = getWeights();
        double[] numericGrad = new double[originalWeights.length];

        // 2) For each weight
        for (int i = 0; i < originalWeights.length; i++) {
            double oldWeight = originalWeights[i];

            // w_i + H
            originalWeights[i] = oldWeight + H;
            setWeights(originalWeights);

            // forward pass on all sequences, sum up
            double fPlus = 0.0;
            for (Sequence seq : sequences) {
                fPlus += forwardPass(seq);
            }

            // w_i - H
            originalWeights[i] = oldWeight - H;
            setWeights(originalWeights);

            // forward pass on all sequences, sum up
            double fMinus = 0.0;
            for (Sequence seq : sequences) {
                fMinus += forwardPass(seq);
            }

            // numeric gradient
            double grad_i = (fPlus - fMinus) / (2.0 * H);
            numericGrad[i] = grad_i;

            // 3) Restore the original weight
            originalWeights[i] = oldWeight;
            setWeights(originalWeights);
        }

        return numericGrad;
    }


    /**
     * This performs a backward pass through the neural network given 
     * outputs from the given instance. This will set the deltas in
     * all the edges and nodes which will be used to calculate the 
     * gradient and perform backpropagation.
     *
     */
    public void backwardPass(Sequence sequence) throws NeuralNetworkException {
        //You need to implement this for Programming Assignment 2 - Part 2
        //note you should start at sequence.getLength() - 2 (not -1) as this is the last
        //time step that was passed forward through the RNN

        // 1) Determine the sequence length
        int sequenceLength;
        if (sequence instanceof CharacterSequence) {
            sequenceLength = ((CharacterSequence)sequence).getLength();
        } else if (sequence instanceof TimeSeries) {
            sequenceLength = ((TimeSeries)sequence).getLength();
        } else {
            throw new NeuralNetworkException("Unknown sequence type for backward pass: " + sequence.getClass().getSimpleName());
        }

        // We only propagated forward up to (sequenceLength - 1) steps,
        // so the last time step to backprop is (sequenceLength - 2).
        // e.g., if sequenceLength = 10, we used time steps 0..8 in forwardPass,
        // so we do backprop from t=8 down to t=0.
        for (int t = sequenceLength - 2; t >= 0; t--) {
            // 2) For each layer in reverse order (output layer down to input)
            for (int layerIndex = layers.length - 1; layerIndex >= 0; layerIndex--) {
                // 3) For each node in that layer
                for (int nodeIndex = 0; nodeIndex < layers[layerIndex].length; nodeIndex++) {
                    RecurrentNode node = layers[layerIndex][nodeIndex];
                    // 4) Let the node handle its own backprop. That calls the edges
                    //    to accumulate weight gradients and pass the delta further back.
                    node.propagateBackward(t);
                }
            }
        }

    }

    /**
     * This gets the gradient of the neural network at its current
     * weights and the given instance using backpropagation (e.g., 
     * the RecurrentNeuralNetwork.backwardPass(Sequence)) Method.
     *
     * Helpful tip: use getDeltas after doing the propagateBackwards through
     * the networks to get the gradients/deltas in the same order as the
     * weights (which will be the same order as they're calculated for
     * the numeric gradient).
     *
     * @param sequence is the training CharacterSequence for the forward and 
     * backward pass.
     */
    public double[] getGradient(Sequence sequence) throws NeuralNetworkException {
        forwardPass(sequence);
        backwardPass(sequence);

        return getDeltas();
    }

    /**
     * This gets the gradient of the neural network at its current
     * weights and the given instance using backpropagation (e.g., 
     * the RecurrentNeuralNetwork.backwardPass(Instance))* Method.
     *
     * Helpful tip: use getDeltas after doing the propagateBackwards through
     * the networks to get the gradients/deltas in the same order as the
     * weights (which will be the same order as they're calculated for
     * the numeric gradient). The resulting gradient should be the sum of
     * each delta for each instance.
     *
     * @param sequences are the training CharacterSequences for the forward and 
     * backward passes.
     */
    public double[] getGradient(List<Sequence> sequences) throws NeuralNetworkException {
        // You need to implement this for Programming Assignment 2 - Part 2

        // 1) Reset all node states and weight deltas to zero
        //reset();

        // 2) For each sequence, do forward + backward passes
        for (Sequence seq : sequences) {
            forwardPass(seq);
            backwardPass(seq);
            // Each backwardPass call accumulates weightDelta in edges/nodes.
            // We do NOT reset after each sequence so that deltas add up.
        }

        // 3) Retrieve the summed weight deltas from the network
        double[] totalGradient = getDeltas();

        // 4) Return the combined gradient
        return totalGradient;
    }
}
