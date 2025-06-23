/**
 * This is a helpful class to let you test the required
 * functionality for Programming Assignment 1 - Part 1.
 *
 */
import java.util.List;

import data.DataSet;
import data.Instance;

import network.LossFunction;
import network.NeuralNetwork;
import network.NeuralNetworkException;

import util.Log;
import util.Vector;


public class PA13GradientDescent {
    public static void helpMessage() {
        Log.info("Usage:");
        Log.info("\tjava PA13GradientDescent <data set> <network type> <gradient descent type> <loss function> <epochs> <bias> <learning rate>");
        Log.info("\t\tdata set can be: 'and', 'or' or 'xor'");
        Log.info("\t\tnetwork type can be: 'tiny', 'small' or 'large'");
        Log.info("\t\tgradient descent type can be: 'stochastic', 'minibatch' or 'batch'");
        Log.info("\t\tloss function can be: 'l1_norm', or 'l2 norm'");
        Log.info("\t\tepochs is an integer > 0");
        Log.info("\t\tbias is a double");
        Log.info("\t\tlearning rate is a double usually small and > 0");
    }

    public static void main(String[] arguments) {
        if (arguments.length != 7) {
            helpMessage();
            System.exit(1);
        }
        String dataSetName = arguments[0];
        String networkType = arguments[1];
        String descentType = arguments[2];
        String lossFunctionName = arguments[3];
        int epochs = Integer.parseInt(arguments[4]);
        double bias = Double.parseDouble(arguments[5]);
        double learningRate = Double.parseDouble(arguments[6]);

        DataSet dataSet = null;
        if (dataSetName.equals("and")) {
            dataSet = new DataSet("and data", "./datasets/and.txt");
        } else if (dataSetName.equals("or")) {
            dataSet = new DataSet("or data", "./datasets/or.txt");
        } else if (dataSetName.equals("xor")) {
            dataSet = new DataSet("xor data", "./datasets/xor.txt");
        } else {
            Log.fatal("unknown data set : " + dataSetName);
            System.exit(1);
        }

        LossFunction lossFunction = LossFunction.NONE;
        if (lossFunctionName.equals("l1_norm")) {
            Log.info("Using an L1_NORM loss function.");
            lossFunction = LossFunction.L1_NORM;
        } else if (lossFunctionName.equals("l2_norm")) {
            Log.info("Using an L2_NORM loss function.");
            lossFunction = LossFunction.L2_NORM;
        } else {
            Log.fatal("unknown loss function : " + lossFunctionName);
            System.exit(1);
        }

        NeuralNetwork nn = null;

        Log.info("Using a tiny neural network.");
        if (networkType.equals("tiny")) {
            Log.info("Using a tiny neural network.");
            nn = PA11Tests.createTinyNeuralNetwork(dataSet, lossFunction);
        } else if (networkType.equals("small")) {
            Log.info("Using a small neural network.");
            nn = PA11Tests.createSmallNeuralNetwork(dataSet, lossFunction);
        } else if (networkType.equals("large")) {
            Log.info("Using a large neural network.");
            nn = PA11Tests.createLargeNeuralNetwork(dataSet, lossFunction);
        } else {
            Log.fatal("unknown network type: " + networkType);
            System.exit(1);
        }


        //start the gradient descent
        try {
            Log.info("Starting " + descentType + " gradient descent!");

            System.out.println(descentType + " " + dataSetName + " " + lossFunctionName + " " + learningRate);

            gradientDescent(descentType, dataSet, nn, epochs, bias, learningRate);
        } catch (NeuralNetworkException e) {
            Log.fatal("gradient descent failed with exception: " + e);
            e.printStackTrace();
            System.exit(1);
        }
    }

    /**
     * This performs one of the three types of gradient descent on a given
     * neural network for a given data set. It will initialize the neural
     * network's weights randomly and set the node's bias to a specified
     * bias. It will run for a specified number of epochs.
     *
     * @param descentType is the type of gradient descent, it can be either "stochastic", "minibatch" or "batch".
     * @param dataSet is the dataSet to train on
     * @param nn is the neural network
     * @param epochs is how many epochs to train the neural network for
     * @param bias is the bias to initialize each node's bias with
     * @param learningRate is the step size/learning rate for the weight updates
     */
    public static void gradientDescent(String descentType, DataSet dataSet, NeuralNetwork nn, int epochs, double bias, double learningRate) throws NeuralNetworkException {
        nn.initializeRandomly(bias);

        Log.info("Initial weights:");
        //Vector.print(nn.getWeights());

        double bestError = 10000;
        for (int i = 0; i < epochs; i++) {

            if (descentType.equals("stochastic")) {
                stochasticEpoch(dataSet, nn, learningRate);

            } else if (descentType.equals("minibatch")) {
                //for now we will just use a batch size of 2 because there are only 
                //4 training samples
                minibatchEpoch(dataSet, nn, 2, learningRate);

            } else if (descentType.equals("batch")) {
                batchEpoch(dataSet, nn, learningRate);

            } else {
                Log.fatal("unknown descent type: " + descentType);
                helpMessage();
                System.exit(1);
            }

            //at the end of each epoch, calculate the error over the entire
            //set of instances and print it out so we can see if we're decreasing
            //the overall error
            double error = nn.forwardPass(dataSet.getInstances());

            if (error < bestError) bestError = error;
            System.out.println(i + " " + bestError + " " + error);
            //Vector.print(nn.getWeights());
            
        }
    }

    public static void stochasticEpoch(DataSet dataSet, NeuralNetwork nn, double learningRate) throws NeuralNetworkException {
        //PA1-3 you need to implement one epoch (pass through the
        //training data) for stochastic gradient descent
        // 1) Shuffle the dataset at the start of the epoch
        dataSet.shuffle();

        // 2) Retrieve the total number of instances
        int numInstances = dataSet.getNumberInstances();

        // 3) For each instance (in shuffled order), compute and apply a weight update
        for (int i = 0; i < numInstances; i++) {
            // a) Grab the instance
            Instance instance = dataSet.getInstance(i);

            // b) Get the gradient for this single sample (forward+backward pass)
            double[] grad = nn.getGradient(instance);

            // c) Retrieve current weights
            double[] weights = nn.getWeights();

            // d) Update the weights immediately
            for (int w = 0; w < weights.length; w++) {
                weights[w] -= learningRate * grad[w];
            }
            nn.setWeights(weights);
        }

    }

    public static void minibatchEpoch(DataSet dataSet, NeuralNetwork nn, int batchSize, double learningRate) throws NeuralNetworkException {
        //PA1-3 you need to implement one epoch (pass through the
        //training data) for minibatch gradient descent
        // 1) Shuffle the dataset for a fresh batch ordering each epoch
        dataSet.shuffle();

        // 2) Basic setup
        int numWeights = nn.getNumberWeights();
        int numInstances = dataSet.getNumberInstances();

        // 3) Loop over the data in chunks of size ‘batchSize’
        for (int start = 0; start < numInstances; start += batchSize) {
            int end = Math.min(start + batchSize, numInstances);

            // a) Accumulate gradients for all samples in the current mini‐batch
            double[] sumGrad = new double[numWeights];
            for (int i = start; i < end; i++) {
                Instance instance = dataSet.getInstance(i);
                double[] grad = nn.getGradient(instance);

                for (int w = 0; w < numWeights; w++) {
                    sumGrad[w] += grad[w];
                }
            }

            // b) Perform one weight update for the entire mini‐batch
            double[] weights = nn.getWeights();
            for (int w = 0; w < numWeights; w++) {
                weights[w] -= learningRate * sumGrad[w];
            }
            nn.setWeights(weights);
        }

    }

    public static void batchEpoch(DataSet dataSet, NeuralNetwork nn, double learningRate) throws NeuralNetworkException {
        //PA1-3 you need to implement one epoch (pass through the training
        //instances) for batch gradient descent
        // 1) Retrieve number of weights and instances
        int numWeights = nn.getNumberWeights();
        int numInstances = dataSet.getNumberInstances();

        // 2) Accumulate the gradient across the entire dataset
        double[] sumGrad = new double[numWeights];
        for (int i = 0; i < numInstances; i++) {
            Instance instance = dataSet.getInstance(i);

            // forward+backward pass to get the gradient for this instance
            double[] grad = nn.getGradient(instance);

            // accumulate the gradients
            for (int w = 0; w < numWeights; w++) {
                sumGrad[w] += grad[w];
            }
        }

        // 3) Perform exactly ONE weight update for the entire dataset
        double[] weights = nn.getWeights();
        for (int w = 0; w < numWeights; w++) {
            weights[w] -= learningRate * sumGrad[w];
        }
        nn.setWeights(weights);
    }
}

