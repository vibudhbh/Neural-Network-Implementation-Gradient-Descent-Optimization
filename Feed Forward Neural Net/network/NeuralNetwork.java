package network;

import java.util.List;

import data.Instance;

import util.Log;


public class NeuralNetwork {
    //this is the loss function for the output of the neural 
    //network, you will use this in PA1-3
    LossFunction lossFunction;
    

    //this is the total number of weights in the neural network
    int numberWeights;
    
    //layers contains all the nodes in the neural network
    Node[][] layers;

    public NeuralNetwork(int inputLayerSize, int[] hiddenLayerSizes, int outputLayerSize, LossFunction lossFunction) {
        this.lossFunction = lossFunction;

        //the number of layers in the neural network is 2 plus the number of hidden layers,
        //one additional for the input, and one additional for the output.

        //create the outer array of the 2-dimensional array of nodes
        layers = new Node[hiddenLayerSizes.length + 2][];
        
        //we will progressively calculate the number of weights as we create the network. the
        //number of edges will be equal to the number of hidden nodes (each has a bias weight, but
        //the input and output nodes do not) plus the number of edges
        numberWeights = 0;

        Log.info("creating a neural network with " + hiddenLayerSizes.length + " hidden layers.");
        for (int layer = 0; layer < layers.length; layer++) {
            
            //determine the layer size depending on the layer number, 0 is the
            //input layer, and the last layer is the output layer, all others
            //are hidden layers
            int layerSize;
            NodeType nodeType;
            ActivationType activationType;
            if (layer == 0) {
                //this is the input layer
                layerSize = inputLayerSize;
                nodeType = NodeType.INPUT;
                activationType = ActivationType.LINEAR;
                Log.info("input layer " + layer + " has " + layerSize + " nodes.");

            } else if (layer < layers.length - 1) {
                //this is a hidden layer
                layerSize = hiddenLayerSizes[layer - 1];
                nodeType = NodeType.HIDDEN;
                activationType = ActivationType.TANH;
                Log.info("hidden layer " + layer + " has " + layerSize + " nodes.");

                //increment the number of weights by the number of nodes in
                //this hidden layer
                numberWeights += layerSize; 
            } else {
                //this is the output layer
                layerSize = outputLayerSize;
                nodeType = NodeType.OUTPUT;
                activationType = ActivationType.SIGMOID;
                Log.info("output layer " + layer + " has " + layerSize + " nodes.");
            }

            //create the layer with the right length and right node types
            layers[layer] = new Node[layerSize];
            for (int j = 0; j < layers[layer].length; j++) {
                layers[layer][j] = new Node(layer, j /*i is the node number*/, nodeType, activationType);
            }
        }
    }

    /**
     * This gets the number of weights in the NeuralNetwork, which should
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
     * This returns an array of every weight (including biases) in the NeuralNetwork.
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
                    throw new NeuralNetworkException("The numberWeights field of the NeuralNetwork was (" + numberWeights + ") but when getting the weights there were more hidden nodes and edges than numberWeights. This should not happen unless numberWeights is not being updated correctly.");
                }
            }
        }

        return weights;
    }

    /**
     * This sets every weight (including biases) in the NeuralNetwork, it sets them in
     * the same order that they are retreived by the getWeights method.
     * This will be very useful in backpropagation and sanity checking. 
     *
     * @throws NeuralNetworkException if numberWeights was not calculated correctly. This shouldn't happen.
     */
    public void setWeights(double[] newWeights) throws NeuralNetworkException {
        if (numberWeights != newWeights.length) {
            throw new NeuralNetworkException("Could not setWeights because the number of new weights: " + newWeights.length + " was not equal to the number of weights in the NeuralNetwork: " + numberWeights);
        }

        int position = 0;
        for (int layer = 0; layer < layers.length; layer++) {
            for (int nodeNumber = 0; nodeNumber < layers[layer].length; nodeNumber++) {
                int nWeights = layers[layer][nodeNumber].setWeights(position, newWeights);
                position += nWeights;

                if (position > numberWeights) {
                    throw new NeuralNetworkException("The numberWeights field of the NeuralNetwork was (" + numberWeights + ") but when setting the weights there were more hidden nodes and edges than numberWeights. This should not happen unless numberWeights is not being updated correctly.");
                }
            }
        }
    }

    /**
     * This returns an array of every delta value (including bias deltas) in the NeuralNetwork (these are the gradients for the biases and weights).
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
                    throw new NeuralNetworkException("The numberWeights field of the NeuralNetwork was (" + numberWeights + ") but when getting the deltas there were more hidden nodes and edges than numberWeights. This should not happen unless numberWeights is not being updated correctly.");
                }
            }
        }

        return deltas;
    }


    /**
     * This adds edges to the NeuralNetwork, connecting each node
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
                    Node inputNode = layers[layer][inputNodeNumber];
                    Node outputNode = layers[layer + 1][outputNodeNumber];
                    new Edge(inputNode, outputNode);

                    //as we added an edge, the number of weights should increase by 1
                    numberWeights++;
                    Log.trace("numberWeights now: " + numberWeights);
                }
            }
        }
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

        // Validating that the layers are within bounds
        if (inputLayer < 0 || inputLayer >= layers.length) {
            throw new NeuralNetworkException("Invalid inputLayer index: " + inputLayer);
        }
        if (outputLayer < 0 || outputLayer >= layers.length) {
            throw new NeuralNetworkException("Invalid outputLayer index: " + outputLayer);
        }

        //  Validating that we're not connecting backwards or within the same layer
        //    We do allow skipping (inputLayer < outputLayer), but not reversing.
        if (inputLayer >= outputLayer) {
            throw new NeuralNetworkException(
                    "Cannot connect layer " + inputLayer + " to layer " + outputLayer +
                            ". outputLayer must be strictly greater than inputLayer."
            );
        }

        // Validating that the node indices are within bounds
        if (inputNumber < 0 || inputNumber >= layers[inputLayer].length) {
            throw new NeuralNetworkException("Invalid inputNumber index: " + inputNumber);
        }
        if (outputNumber < 0 || outputNumber >= layers[outputLayer].length) {
            throw new NeuralNetworkException("Invalid outputNumber index: " + outputNumber);
        }

        // Creating a new edge from the specified input node to the specified output node
        Node inputNode = layers[inputLayer][inputNumber];
        Node outputNode = layers[outputLayer][outputNumber];
        new Edge(inputNode, outputNode);

        // Updating the number of weights in the neural network
        numberWeights++;
        Log.trace("Connected node (" + inputLayer + "," + inputNumber + ") to "
                + "(" + outputLayer + "," + outputNumber + "). numberWeights now: " + numberWeights);
    }

    /**
     * This initializes the weights properly by setting the incoming
     * weights for each edge using a random normal distribution (i.e.,
     * a gaussian distribution) and dividing the randomly generated
     * weight by sqrt(n) where n is the fan-in of the node. It also
     * sets the bias for each node to the given parameter.
     *
     * For example, if we have a node N which has 5 input edges,
     * the weights of each of those edges will be generated by
     * Random.nextGaussian()/sqrt(5). The best way to do this is
     * to iterate over each n to set the bias of each node to.
     */
    public void initializeRandomly(double bias) {
        //You need to implement this for PA1-3
        for (int layerIndex = 1; layerIndex < layers.length; layerIndex++) {
            for (Node node : layers[layerIndex]) {
                // This method is implemented in Node.java
                node.initializeWeightsAndBias(bias);
            }
        }
    }



    /**
     * This performs a forward pass through the neural network given
     * inputs from the input instance.
     *
     * @param instance is the data set instance to pass through the network
     *
     * @return the sum of the output of all output nodes
     */
    public double forwardPass(Instance instance) throws NeuralNetworkException {
        //be sure to reset before doing a forward pass
        reset();

        // Set the input layer's activation from instance.inputs
        if (instance.inputs.length != layers[0].length) {
            throw new NeuralNetworkException(
                    "Mismatch: instance has " + instance.inputs.length
                            + " inputs, but input layer has " + layers[0].length + " nodes."
            );
        }
        for (int i = 0; i < layers[0].length; i++) {
            Node inputNode = layers[0][i];
            // For an input node, we just store the input in preActivationValue
            inputNode.preActivationValue = instance.inputs[i];
            // Then a linear activation sets postActivationValue = preActivationValue
            inputNode.applyLinear();
        }

        //    "Propagate Forward" layer by layer
        //    We start at the input layer (layer 0) and go up to the last layer.
        //    Each call to node.propagateForward() will add its postActivationValue
        //    (after applying its activation) to the next layer's preActivationValue.
        for (int layerIndex = 0; layerIndex < layers.length; layerIndex++) {
            for (Node node : layers[layerIndex]) {
                // Each node handles its own bias (if hidden)
                // and pushes to the next layer's preActivationValue
                node.propagateForward();
            }
        }

        //The following is needed for PA1-3 and PA1-4
        int outputLayer = layers.length - 1;
        int nOutputs = layers[outputLayer].length;

        if (lossFunction == LossFunction.NONE) {
            double outputSum = 0;
            for (int number = 0; number < nOutputs; number++) {
                Node outputNode = layers[outputLayer][number];
                outputSum += outputNode.postActivationValue;

                outputNode.delta = 1;
            }

            return outputSum; //just return the sum of the outputs if we have no loss function

        } else if (lossFunction == LossFunction.L1_NORM) {
            //Implement this for PA1-3
            double loss = 0.0;
            for (int j = 0; j < nOutputs; j++) {
                Node outputNode = layers[outputLayer][j];
                double prediction = outputNode.postActivationValue;
                double target = instance.expectedOutputs[j];

                double diff = prediction - target;
                loss += Math.abs(diff);

                // set the node's delta (the derivative)
                if (diff > 0) {
                    outputNode.delta = 1.0;
                } else if (diff < 0) {
                    outputNode.delta = -1.0;
                } else {
                    outputNode.delta = 0.0;
                }
            }
            return loss; //return the calculated l1 norm loss

        } else if (lossFunction == LossFunction.L2_NORM) {
            //Implement this for PA1-3
            double sumSquares = 0.0;

            // 1) Accumulate sum of squares
            for (int j = 0; j < nOutputs; j++) {
                Node outputNode = layers[outputLayer][j];
                double prediction = outputNode.postActivationValue;  // z_j
                double target = instance.expectedOutputs[j];                 // y_j
                double diff = (prediction - target);
                sumSquares += diff * diff;
            }

            // 2) Take the square root for the final L2 loss
            double l2norm = Math.sqrt(sumSquares);

            // Edge case: avoid dividing by zero if all predictions match targets
            // or the sumSquares is extremely small:
            if (l2norm < 1e-12) {
                // e.g., set the loss to 0.0, and deltas to 0
                for (int j = 0; j < nOutputs; j++) {
                    layers[outputLayer][j].delta = 0.0;
                }
                return 0.0;
            }

            // 3) Each output node’s delta = (z_j - y_j) / l2norm
            for (int j = 0; j < nOutputs; j++) {
                Node outputNode = layers[outputLayer][j];
                double prediction = outputNode.postActivationValue;
                double target = instance.expectedOutputs[j];
                double diff = (prediction - target);

                // derivative wrt z_j
                outputNode.delta = diff / l2norm;
            }

            // 4) Return the actual L2 value
            return l2norm; //return the calculated l2 norm loss

        } else if (lossFunction == LossFunction.SVM) {
            //Implement this for PA1-4

            // The single label is stored in expectedOutputs[0], e.g. 0,1,2 for iris
            int y = (int) instance.expectedOutputs[0];

            // The array to store partial derivatives w.r.t. postActivations
            // We'll accumulate these, then set outputNodes[j].delta = dScores[j].
            double[] dScores = new double[nOutputs];
            // Initialize derivative array to 0
            for (int i = 0; i < dScores.length; i++) {
                dScores[i] = 0.0;
            }

            // Grab each output's postActivation
            double[] z = new double[nOutputs];
            for (int j = 0; j < nOutputs; j++) {
                z[j] = layers[outputLayer][j].postActivationValue;
            }
            double z_y = z[y];

            double loss = 0.0;
            // For each j != y, compute margin = z_j - z_y + 1
            for (int j = 0; j < nOutputs; j++) {
                if (j == y) {
                    continue; // skip the correct class in sum
                }
                double margin = z[j] - z_y + 1.0;
                if (margin > 0.0) {
                    loss += margin;
                    dScores[j] += 1.0;   // derivative wrt z_j
                    dScores[y] -= 1.0;   // derivative wrt z_y
                }
            }

            // Now store these derivatives into each Node's delta
            for (int j = 0; j < nOutputs; j++) {
                Node out = layers[outputLayer][j];
                out.delta = dScores[j];
            }

            return loss;

        } else if (lossFunction == LossFunction.SOFTMAX) {
            //Implement this for PA1-4

            // 1) Compute exponentials of each output node's activation
            double[] exps = new double[nOutputs];
            double sumExp = 0.0;
            for (int j = 0; j < nOutputs; j++) {
                Node outputNode = layers[outputLayer][j];
                double z_j = outputNode.postActivationValue;
                exps[j] = Math.exp(z_j);
                sumExp += exps[j];
            }

            // Convert single label to an integer index
            int classIndex = (int) instance.expectedOutputs[0];

            // 2) Compute probabilities and cross-entropy
            double loss = 0.0;
            for (int j = 0; j < nOutputs; j++) {
                Node outputNode = layers[outputLayer][j];
                double p_j = exps[j] / sumExp; // softmax probability

                // If j is the correct class => target=1, else => 0
                double target = (j == classIndex) ? 1.0 : 0.0;

                // cross-entropy component: -log(p_j) if target=1
                if (target > 0.5) {
                    // clamp for safety
                    loss -= Math.log(Math.max(p_j, 1e-12));
                }

                // gradient wrt z_j = p_j - target
                outputNode.delta = p_j - target;
            }

            return loss; // negative log-likelihood

        } else {
            throw new NeuralNetworkException("Could not do forward pass on NeuralNetwork because lossFunction was unknown: " + lossFunction);
        }
    }

    /**
     * This performs multiple forward passes through the neural network
     * by multiple instances are returns the output sum.
     *
     * @param instances is the set of instances to pass through the network
     *
     * @return the sum of their outputs
     */
    public double forwardPass(List<Instance> instances) throws NeuralNetworkException {
        double sum = 0.0;

        for (Instance instance : instances) {
            sum += forwardPass(instance);
        }

        return sum;
    }

    /**
     * This performs multiple forward passes through the neural network
     * and calculates how many of the instances were classified correctly.
     *
     * @param instances is the set of instances to pass through the network
     *
     * @return a percentage (between 0 and 1) of how many instances were
     * correctly classified
     */
    public double calculateAccuracy(List<Instance> instances) throws NeuralNetworkException {
        //need to implement this for PA1-4
        //the output node with the maximum value is the predicted class
        //you need to sum up how many of these match the actual class
        //from the instance, and then calculate: num correct / total
        //to get a percentage accuracy
        int correct = 0;
        int total = instances.size();

        for (Instance instance : instances) {
            // 1) Forward pass
            forwardPass(instance);

            // 2) Identify which output node has the highest postActivationValue
            int predictedClass = -1;
            double bestScore = Double.NEGATIVE_INFINITY;

            // The output layer is layers[layers.length - 1]
            Node[] outputNodes = layers[layers.length - 1];
            for (int j = 0; j < outputNodes.length; j++) {
                double score = outputNodes[j].postActivationValue;
                if (score > bestScore) {
                    bestScore = score;
                    predictedClass = j;
                }
            }

            // 3) Compare predictedClass with the actual class index
            int actualClass = (int) instance.expectedOutputs[0];
            if (predictedClass == actualClass) {
                correct++;
            }
        }

        // 4) Return percentage
        return (double) correct / total;
    }


    /**
     * This gets the output values of the neural network 
     * after a forward pass.
     *
     * @return an array of the output values from this neural network
     */
    public double[] getOutputValues() {
        //the number of output values is the number of output nodes
        int outputLayer = layers.length - 1;
        int nOutputs = layers[outputLayer].length;

        double[] outputValues = new double[nOutputs];

        for (int number = 0; number < nOutputs; number++) {
            outputValues[number] = layers[outputLayer][number].postActivationValue;
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
    public double[] getNumericGradient(Instance instance) throws NeuralNetworkException {
        //You need to implement this for Programming Assignment 1 - Part 2

        // Retrieve the current weights
        double[] weights = getWeights();
        double[] gradients = new double[weights.length];

        // Iterate over each weight
        for (int i = 0; i < weights.length; i++) {
            // Store the original weight
            double originalWeight = weights[i];

            // Calculate f(x + H)
            weights[i] = originalWeight + H;
            setWeights(weights); // Update the weights in the network
            double fxPlusH = forwardPass(instance);

            // Calculate f(x - H)
            weights[i] = originalWeight - H;
            setWeights(weights); // Update the weights in the network
            double fxMinusH = forwardPass(instance);

            // Compute the gradient using the central difference formula
            gradients[i] = (fxPlusH - fxMinusH) / (2 * H);

            // Restore the original weight
            weights[i] = originalWeight;
        }

        // Restore the original weights in the network
        setWeights(weights);

        return gradients;
    }

    /**
     * This calculates the gradient of the neural network with it's current
     * weights for a given DataSet Instance using the finite difference method:
     * gradient[i] = (f(x where x[i] = x[i] + H) - f(x where x[i] = x[i] - H)) / 2h
     */
    public double[] getNumericGradient(List<Instance> instances) throws NeuralNetworkException {
        //You need to implement this for Programming Assignment 1 - Part 2
        // Retrieve the current weights
        double[] weights = getWeights();
        double[] gradients = new double[weights.length];

        // Iterate over each weight
        for (int i = 0; i < weights.length; i++) {
            // Store the original weight
            double originalWeight = weights[i];

            // Calculate f(x + H) over all instances
            weights[i] = originalWeight + H;
            setWeights(weights); // Update the weights in the network
            double fxPlusH = forwardPass(instances);

            // Calculate f(x - H) over all instances
            weights[i] = originalWeight - H;
            setWeights(weights); // Update the weights in the network
            double fxMinusH = forwardPass(instances);

            // Compute the gradient using the central difference formula
            gradients[i] = (fxPlusH - fxMinusH) / (2 * H);

            // Restore the original weight
            weights[i] = originalWeight;
        }

        // Restore the original weights in the network
        setWeights(weights);

        return gradients;
    }


    /**
     * This performs a backward pass through the neural network given
     * the deltas assigned to the output nodes which were calculated in
     * the last call to the forwardPass method. At the end of this method
     * all weightDelta and biasDelta values in the neural network should
     * be computed so they can be returned with the getGradient method.
     */
    public void backwardPass() throws NeuralNetworkException {
        // You need to implement this for Programming Assignment 1 - Part 2

        // Start from the last layer (output layer) and move backwards
        // through the hidden layers. We stop at the first hidden layer
        // (layer index = 1) because the input layer (layer = 0) does not
        // receive any backprop delta (it has no incoming edges).
        for (int layerIndex = layers.length - 1; layerIndex > 0; layerIndex--) {
            for (Node node : layers[layerIndex]) {
                // Each node takes its own delta and backpropagates it
                // to its incoming edges.
                node.propagateBackward();
            }
        }
    }

    /**
     * This gets the gradient of the neural network at its current
     * weights and the given instance using backpropagation (e.g., 
     * the NeuralNetwork.backwardPass(Instance))* Method.
     *
     * Helpful tip: use getDeltas after doing the propagateBackwards through
     * the networks to get the gradients/deltas in the same order as the
     * weights (which will be the same order as they're calculated for
     * the numeric gradient).
     *
     * @param instance is the training instance/sample for the forward and 
     * backward pass.
     */
    public double[] getGradient(Instance instance) throws NeuralNetworkException {
        forwardPass(instance);
        backwardPass();

        return getDeltas();
    }

    /**
     * This gets the gradient of the neural network at its current
     * weights and the given instance using backpropagation (e.g., 
     * the NeuralNetwork.backwardPass(Instance))* Method.
     *
     * Helpful tip: use getDeltas after doing the propagateBackwards through
     * the networks to get the gradients/deltas in the same order as the
     * weights (which will be the same order as they're calculated for
     * the numeric gradient). The resulting gradient should be the sum of
     * each delta for each instance.
     *
     * @param instances are the training instances/samples for the forward and 
     * backward passes.
     */
    public double[] getGradient(List<Instance> instances) throws NeuralNetworkException {
        // You need to implement this for Programming Assignment 1 - Part 2

        // 1) Capture the current weights
        double[] originalWeights = getWeights();

        // 2) Initialize an array to accumulate gradients
        double[] accumulatedGradient = new double[numberWeights];

        // 3) Loop over each instance
        for (Instance instance : instances) {
            // a) Restore original weights for a fresh start
            setWeights(originalWeights);

            // b) Reset the network’s stored values/deltas
            reset();

            // c) Forward pass
            forwardPass(instance);

            // d) Backward pass
            backwardPass();

            // e) Get the deltas for this single instance
            double[] instanceDeltas = getDeltas();

            // f) Accumulate them
            for (int i = 0; i < numberWeights; i++) {
                accumulatedGradient[i] += instanceDeltas[i];
            }
        }

        // 4) Restore original weights
        setWeights(originalWeights);

        // 5) Return the sum of gradients over all instances
        return accumulatedGradient;
    }
}
