/**
 * This is a helpful class to let you test the required
 * functionality for Programming Assignment 1 - Part 1.
 *
 */
import java.util.Arrays;
import java.util.List;

import data.DataSet;
import data.Instance;

import network.LossFunction;
import network.NeuralNetwork;
import network.NeuralNetworkException;

import util.Log;
import util.Vector;


public class PA14GradientDescent {
    public static void helpMessage() {
        Log.info("Usage:");
        Log.info("\tjava PA13GradientDescent <data set> <gradient descent type> <batch size> <loss function> <epochs> <bias> <learning rate> <mu> <layer_size_1 ... layer_size_n");
        Log.info("\t\tdata set can be: 'and', 'or' or 'xor', 'iris' or 'mushroom'");
        Log.info("\t\tgradient descent type can be: 'stochastic', 'minibatch' or 'batch'");
        Log.info("\t\tbatch size should be > 0. Will be ignored for stochastic or batch gradient descent");
        Log.info("\t\tloss function can be: 'l1_norm', 'l2_norm', 'svm' or 'softmax'");
        Log.info("\t\tepochs is an integer > 0");
        Log.info("\t\tbias is a double");
        Log.info("\t\tlearning rate is a double usually small and > 0");
        Log.info("\t\tmu is a double < 1 and typical values are 0.5, 0.9, 0.95 and 0.99");
        Log.info("\t\toptimizer type can be: 'nesterov', 'rmsprop' or 'adam'");
        Log.info("\t\tlayer_size_1..n is a list of integers which are the number of nodes in each hidden layer");

    }

    public static void main(String[] arguments) {
        if (arguments.length < 9) {
            helpMessage();
            System.exit(1);
        }

        String dataSetName = arguments[0];
        String descentType = arguments[1];
        int batchSize = Integer.parseInt(arguments[2]);
        String lossFunctionName = arguments[3];
        int epochs = Integer.parseInt(arguments[4]);
        double bias = Double.parseDouble(arguments[5]);
        double learningRate = Double.parseDouble(arguments[6]);
        double mu = Double.parseDouble(arguments[7]);
        String optimizerType = arguments[8]; // Now taken from the command line

        int[] layerSizes = new int[arguments.length - 9]; // the remaining arguments are the layer sizes
        for (int i = 9; i < arguments.length; i++) {
            layerSizes[i - 9] = Integer.parseInt(arguments[i]);
        }

        //the and, or and xor datasets will have 1 output (the number of output columns)
        //but the iris and mushroom datasets will have the number of output classes
        int outputLayerSize = 0;

        DataSet dataSet = null;
        if (dataSetName.equals("and")) {
            dataSet = new DataSet("and data", "./datasets/and.txt");
            outputLayerSize = dataSet.getNumberOutputs();
        } else if (dataSetName.equals("or")) {
            dataSet = new DataSet("or data", "./datasets/or.txt");
            outputLayerSize = dataSet.getNumberOutputs();
        } else if (dataSetName.equals("xor")) {
            dataSet = new DataSet("xor data", "./datasets/xor.txt");
            outputLayerSize = dataSet.getNumberOutputs();
        } else if (dataSetName.equals("iris")) {
            //TODO: PA1-4: Make sure you implement the getInputMeans,
            //getInputStandardDeviations and normalize methods in
            //DataSet to get this to work.
            dataSet = new DataSet("iris data", "./datasets/iris.txt");
            double[] means = dataSet.getInputMeans();
            double[] stdDevs = dataSet.getInputStandardDeviations();
            Log.info("data set means: " + Arrays.toString(means));
            Log.info("data set standard deviations: " + Arrays.toString(stdDevs));
            dataSet.normalize(means, stdDevs);

            outputLayerSize = dataSet.getNumberClasses();
        } else if (dataSetName.equals("mushroom")) {
            dataSet = new DataSet("mushroom data", "./datasets/agaricus-lepiota.txt");
            outputLayerSize = dataSet.getNumberClasses();
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
        } else if (lossFunctionName.equals("svm")) {
            Log.info("Using an SVM loss function.");
            lossFunction = LossFunction.SVM;
        } else if (lossFunctionName.equals("softmax")) {
            Log.info("Using an SOFTMAX loss function.");
            lossFunction = LossFunction.SOFTMAX;
        } else {
            Log.fatal("unknown loss function : " + lossFunctionName);
            System.exit(1);
        }

        NeuralNetwork nn = new NeuralNetwork(dataSet.getNumberInputs(), layerSizes, outputLayerSize, lossFunction);
        try {
            nn.connectFully();
        } catch (NeuralNetworkException e) {
            Log.fatal("ERROR connecting the neural network -- this should not happen!.");
            e.printStackTrace();
            System.exit(1);
        }

        //start the gradient descent
        try {
            Log.info("Starting " + descentType + " gradient descent!");
            if (descentType.equals("minibatch")) {
                Log.info(descentType + "(" + batchSize + "), " + dataSetName + ", " + lossFunctionName + ", lr: " + learningRate + ", mu:" + mu + ", optimizer: " + optimizerType);
            } else {
                Log.info(descentType + ", " + dataSetName + ", " + lossFunctionName + ", lr: " + learningRate + ", mu:" + mu + ", optimizer: " + optimizerType);
            }

            nn.initializeRandomly(bias);
            int weightCount = nn.getNumberWeights();

            //TODO: For PA1-4 use this and implement nesterov momentum
            //java will initialize each element in the array to 0
            double[] weights = nn.getWeights();
            double[] velocity = new double[weightCount];

            //TODO: BONUS PA1-4: (1 point) implement the RMSprop 
            //per-parameter adaptive learning rate method.
            double[] gradSquare = new double[weightCount];
            double decayRate = 0.9; // typical RMSProp default
            double epsilon = 1e-8;

            //TODO: BONUS PA1-4: (1 point) implement the Adam
            //per-parameter adaptive learning rate method.
            //For these you will need to add a command line flag
            //to select if which method you'll use (nesterov, rmsprop or adam)
            double[] m = new double[weights.length];  // first moment
            double[] v = new double[weights.length];  // second moment
            double beta1 = 0.9;  // typical default
            double beta2 = 0.999;
            double adamEpsilon = 1e-8;
            int t = 0; // Adam iteration counter

            double bestError = 10000;
            double error = nn.forwardPass(dataSet.getInstances()) / dataSet.getNumberInstances();
            double accuracy = nn.calculateAccuracy(dataSet.getInstances());

            if (error < bestError) bestError = error;
            System.out.println("  " + bestError + " " + error + " " + String.format("%10.5f", accuracy * 100.0) /*make hte accuracy a percentage*/);

            for (int epc = 0; epc < epochs; epc++) {

                if (descentType.equals("stochastic")) {
                    //TODO: PA1-4 you need to implement one epoch (pass through the
                    //training data) for stochastic gradient descent
                    // --- Stochastic Gradient Descent ---
                    dataSet.shuffle();
                    int numInstances = dataSet.getNumberInstances();
                    for (int j = 0; j < numInstances; j++) {
                        Instance instance = dataSet.getInstance(j);
                        double[] grad = nn.getGradient(instance);

                        switch (optimizerType.toLowerCase()) {
                            case "nesterov": {
                                // Look-ahead: temporarily update weights
                                for (int i = 0; i < weightCount; i++) {
                                    weights[i] += mu * velocity[i];
                                }
                                nn.setWeights(weights);
                                // Recompute gradient from look-ahead position
                                grad = nn.getGradient(instance);
                                // Update velocity and weights
                                for (int i = 0; i < weightCount; i++) {
                                    velocity[i] = mu * velocity[i] - learningRate * grad[i];
                                    weights[i] += velocity[i];
                                }
                                nn.setWeights(weights);
                                break;
                            }
                            case "rmsprop": {
                                for (int i = 0; i < weightCount; i++) {
                                    gradSquare[i] = decayRate * gradSquare[i] + (1 - decayRate) * (grad[i] * grad[i]);
                                    double rms = Math.sqrt(gradSquare[i]) + epsilon;
                                    velocity[i] = mu * velocity[i] - (learningRate / rms) * grad[i];
                                    weights[i] += velocity[i];
                                }
                                nn.setWeights(weights);
                                break;
                            }
                            case "adam": {
                                t++;
                                for (int i = 0; i < weightCount; i++) {
                                    m[i] = beta1 * m[i] + (1 - beta1) * grad[i];
                                    v[i] = beta2 * v[i] + (1 - beta2) * (grad[i] * grad[i]);
                                    double mHat = m[i] / (1 - Math.pow(beta1, t));
                                    double vHat = v[i] / (1 - Math.pow(beta2, t));
                                    double deltaW = -learningRate * (mHat / (Math.sqrt(vHat) + adamEpsilon));
                                    weights[i] += deltaW;
                                }
                                nn.setWeights(weights);
                                break;
                            }
                            default: {
                                // Plain stochastic gradient descent update
                                for (int i = 0; i < weightCount; i++) {
                                    weights[i] -= learningRate * grad[i];
                                }
                                nn.setWeights(weights);
                                break;
                            }
                        }
                    }


                } else if (descentType.equals("minibatch")) {
                    //TODO: PA1-4 you need to implement one epoch (pass through the
                    //training data) for minibatch gradient descent
                    dataSet.shuffle();
                    int numInstances = dataSet.getNumberInstances();
                    for (int start = 0; start < numInstances; start += batchSize) {
                        int end = Math.min(start + batchSize, numInstances);
                        double[] batchGrad = new double[weightCount];
                        // Initialize batchGrad to 0:
                        for (int i = 0; i < weightCount; i++) {
                            batchGrad[i] = 0.0;
                        }
                        // Accumulate gradient over the mini-batch:
                        for (int i = start; i < end; i++) {
                            Instance instance = dataSet.getInstance(i);
                            double[] grad = nn.getGradient(instance);
                            for (int k = 0; k < weightCount; k++) {
                                batchGrad[k] += grad[k];
                            }
                        }
                        // Optionally, average the gradient:
                        for (int i = 0; i < weightCount; i++) {
                            batchGrad[i] /= (end - start);
                        }
                        // Now apply the optimizer update using batchGrad:
                        switch (optimizerType.toLowerCase()) {
                            case "nesterov": {
                                for (int i = 0; i < weightCount; i++) {
                                    weights[i] += mu * velocity[i];
                                }
                                nn.setWeights(weights);
                                // Recompute gradient over the mini-batch:
                                double[] newBatchGrad = new double[weightCount];
                                for (int i = start; i < end; i++) {
                                    Instance instance = dataSet.getInstance(i);
                                    double[] grad = nn.getGradient(instance);
                                    for (int k = 0; k < weightCount; k++) {
                                        newBatchGrad[k] += grad[k];
                                    }
                                }
                                for (int i = 0; i < weightCount; i++) {
                                    newBatchGrad[i] /= (end - start);
                                }
                                batchGrad = newBatchGrad;
                                for (int i = 0; i < weightCount; i++) {
                                    velocity[i] = mu * velocity[i] - learningRate * batchGrad[i];
                                    weights[i] += velocity[i];
                                }
                                nn.setWeights(weights);
                                break;
                            }
                            case "rmsprop": {
                                for (int i = 0; i < weightCount; i++) {
                                    gradSquare[i] = decayRate * gradSquare[i] + (1 - decayRate) * (batchGrad[i] * batchGrad[i]);
                                    double rms = Math.sqrt(gradSquare[i]) + epsilon;
                                    velocity[i] = mu * velocity[i] - (learningRate / rms) * batchGrad[i];
                                    weights[i] += velocity[i];
                                }
                                nn.setWeights(weights);
                                break;
                            }
                            case "adam": {
                                t++;
                                for (int i = 0; i < weightCount; i++) {
                                    m[i] = beta1 * m[i] + (1 - beta1) * batchGrad[i];
                                    v[i] = beta2 * v[i] + (1 - beta2) * (batchGrad[i] * batchGrad[i]);
                                    double mHat = m[i] / (1 - Math.pow(beta1, t));
                                    double vHat = v[i] / (1 - Math.pow(beta2, t));
                                    double deltaW = -learningRate * (mHat / (Math.sqrt(vHat) + adamEpsilon));
                                    weights[i] += deltaW;
                                }
                                nn.setWeights(weights);
                                break;
                            }
                            default: {
                                for (int i = 0; i < weightCount; i++) {
                                    weights[i] -= learningRate * batchGrad[i];
                                }
                                nn.setWeights(weights);
                                break;
                            }
                        }
                    }

                } else if (descentType.equals("batch")) {
                    //TODO: PA1-4 you need to implement one epoch (pass through the training
                    //instances) for batch gradient descent
                    // --- Batch Gradient Descent: use the entire dataset ---
                    int numInstances = dataSet.getNumberInstances();
                    double[] batchGrad = new double[weightCount];
                    // Initialize batchGrad to 0:
                    for (int i = 0; i < weightCount; i++) {
                        batchGrad[i] = 0.0;
                    }
                    // Accumulate gradient over the entire dataset:
                    for (int i = 0; i < numInstances; i++) {
                        Instance instance = dataSet.getInstance(i);
                        double[] grad = nn.getGradient(instance);
                        for (int j = 0; j < weightCount; j++) {
                            batchGrad[j] += grad[j];
                        }
                    }
                    // Optionally average:
                    for (int i = 0; i < weightCount; i++) {
                        batchGrad[i] /= numInstances;
                    }
                    switch (optimizerType.toLowerCase()) {
                        case "nesterov": {
                            for (int i = 0; i < weightCount; i++) {
                                weights[i] += mu * velocity[i];
                            }
                            nn.setWeights(weights);
                            batchGrad = nn.getGradient(dataSet.getInstances());
                            for (int i = 0; i < weightCount; i++) {
                                velocity[i] = mu * velocity[i] - learningRate * batchGrad[i];
                                weights[i] += velocity[i];
                            }
                            nn.setWeights(weights);
                            break;
                        }
                        case "rmsprop": {
                            for (int i = 0; i < weightCount; i++) {
                                gradSquare[i] = decayRate * gradSquare[i] + (1 - decayRate) * (batchGrad[i] * batchGrad[i]);
                                double rms = Math.sqrt(gradSquare[i]) + epsilon;
                                velocity[i] = mu * velocity[i] - (learningRate / rms) * batchGrad[i];
                                weights[i] += velocity[i];
                            }
                            nn.setWeights(weights);
                            break;
                        }
                        case "adam": {
                            t++;
                            for (int i = 0; i < weightCount; i++) {
                                m[i] = beta1 * m[i] + (1 - beta1) * batchGrad[i];
                                v[i] = beta2 * v[i] + (1 - beta2) * (batchGrad[i] * batchGrad[i]);
                                double mHat = m[i] / (1 - Math.pow(beta1, t));
                                double vHat = v[i] / (1 - Math.pow(beta2, t));
                                double deltaW = -learningRate * (mHat / (Math.sqrt(vHat) + adamEpsilon));
                                weights[i] += deltaW;
                            }
                            nn.setWeights(weights);
                            break;
                        }
                        default: {
                            for (int i = 0; i < weightCount; i++) {
                                weights[i] -= learningRate * batchGrad[i];
                            }
                            nn.setWeights(weights);
                            break;
                        }
                    }

                } else {
                    Log.fatal("unknown descent type: " + descentType);
                    helpMessage();
                    System.exit(1);
                }

                //Log.info("weights: " + Arrays.toString(nn.getWeights()));

                //at the end of each epoch, calculate the error over the entire
                //set of instances and print it out so we can see if we're decreasing
                //the overall error
                error = nn.forwardPass(dataSet.getInstances()) / dataSet.getNumberInstances();
                accuracy = nn.calculateAccuracy(dataSet.getInstances());

                if (error < bestError) bestError = error;
                System.out.println(epc + " " + bestError + " " + error + " " + String.format("%10.5f", accuracy * 100.0) /*make hte accuracy a percentage*/);
            }

        } catch (NeuralNetworkException e) {
            Log.fatal("gradient descent failed with exception: " + e);
            e.printStackTrace();
            System.exit(1);
        }
    }
}

