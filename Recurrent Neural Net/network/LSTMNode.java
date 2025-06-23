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

import util.Log;

public class LSTMNode extends RecurrentNode {

    //these are the weight values for the LSTM node
    double wi;
    double wf;
    double wc;
    double wo;

    double ui;
    double uf;
    double uo;

    //these are the bias values for the LSTM node
    double bf;
    double bi;
    double bc;
    double bo;

    //these are the deltas for the weights and biases
    public double delta_wi;
    public double delta_wf;
    public double delta_wc;
    public double delta_wo;

    public double delta_ui;
    public double delta_uf;
    public double delta_uo;

    public double delta_bf;
    public double delta_bi;
    public double delta_bc;
    public double delta_bo;

    //this is the delta value for ct in the diagram, it will be
    //set to the sum of the delta coming in from the outputs (delta)
    //times ot for the time step, plus whatever deltas came in from
    //the subsequent time step during backprop
    public double[] delta_ct;

    //input gate values for each time step
    public double[] it;

    //forward gate values for each time step
    public double[] ft;

    //cell values for each time step
    public double[] ct;

    //output gate values for each time step
    public double[] ot;

    //variable C saved for doing the backward pass
    public double[] C;


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
    public LSTMNode(int layer, int number, NodeType nodeType, int maxSequenceLength) {
        super(layer, number, nodeType, maxSequenceLength, null);

        delta_ct = new double[maxSequenceLength];

        ft = new double[maxSequenceLength];
        it = new double[maxSequenceLength];
        ot = new double[maxSequenceLength];
        ct = new double[maxSequenceLength];
        C = new double[maxSequenceLength];
    }

    /**
     * This resets the values which need to be recalcualted for
     * each forward and backward pass. It will also reset the
     * deltas for outgoing nodes.
     */
    public void reset() {
        //use RecurrentNode's reset to reset everything this has inherited from
        //RecurrentNode, then reset the LSTMNode's fields
        super.reset();
        Log.trace("Resetting LSTM node: " + toString());

        for (int timeStep = 0; timeStep < maxSequenceLength; timeStep++) {
            ft[timeStep] = 0;
            it[timeStep] = 0;
            ot[timeStep] = 0;
            ct[timeStep] = 0;
            C[timeStep] = 0;
            delta_ct[timeStep] = 0;
        }

        delta_wi = 0;
        delta_wf = 0;
        delta_wo = 0;
        delta_wc = 0;

        delta_ui = 0;
        delta_uf = 0;
        delta_uo = 0;

        delta_bi = 0;
        delta_bf = 0;
        delta_bo = 0;
        delta_bc = 0;
    }


    /**
     * We need to override the getWeightNames from RecurrentNode as
     * an LSTMNode will have 11 weight and bias names as opposed to
     * just one bias.
     *
     * @param position is the index to start setting weights in the weights parameter
     * @param weightNames is the array of weight nameswe're setting.
     *
     * @return the number of weights set in the weights parameter
     */
    public int getWeightNames(int position, String[] weightNames) {
        int weightCount = 0;

        //the first weight set will be the bias if it is a hidden node
        if (nodeType != NodeType.INPUT) {
            weightNames[position] = "LSTM Node [layer " + layer + ", number " + number + ", wi]";
            weightNames[position + 1] = "LSTM Node [Layer " + layer + ", number " + number + ", wf]";
            weightNames[position + 2] = "LSTM Node [Layer " + layer + ", number " + number + ", wc]";
            weightNames[position + 3] = "LSTM Node [Layer " + layer + ", number " + number + ", wo]";

            weightNames[position + 4] = "LSTM Node [Layer " + layer + ", number " + number + ", ui]";
            weightNames[position + 5] = "LSTM Node [Layer " + layer + ", number " + number + ", uf]";
            weightNames[position + 6] = "LSTM Node [Layer " + layer + ", number " + number + ", uo]";

            weightNames[position + 7] = "LSTM Node [Layer " + layer + ", number " + number + ", bi]";
            weightNames[position + 8] = "LSTM Node [Layer " + layer + ", number " + number + ", bf]";
            weightNames[position + 9] = "LSTM Node [Layer " + layer + ", number " + number + ", bc]";
            weightNames[position + 10] = "LSTM Node [Layer " + layer + ", number " + number + ", bo]";

            weightCount += 11;
        }

        for (Edge edge : outputEdges) {
            String targetType = "";
            if (edge.outputNode instanceof LSTMNode) targetType = "LSTM ";
            weightNames[position + weightCount] = "Edge from LSTM Node [layer " + layer + ", number " + number + "] to " + targetType + "Node [layer " + edge.outputNode.layer + ", number " + edge.outputNode.number + "]";
            weightCount++;
        }

        for (RecurrentEdge recurrentEdge : outputRecurrentEdges) {
            String targetType = "";
            if (recurrentEdge.outputNode instanceof LSTMNode) targetType = "LSTM ";

            weightNames[position + weightCount] = "Recurrent Edge from LSTM Node [layer " + layer + ", number " + number + "] to " + targetType + "Node [layer " + recurrentEdge.outputNode.layer + ", number " + recurrentEdge.outputNode.number + "]";
            weightCount++;
        }


        return weightCount;
    }



    /**
     * We need to override the getWeights from RecurrentNode as
     * an LSTMNode will have 11 weights and biases as opposed to
     * just one bias.
     *
     * @param position is the index to start setting weights in the weights parameter
     * @param weights is the array of weights we're setting.
     *
     * @return the number of weights set in the weights parameter
     */
    public int getWeights(int position, double[] weights) {
        int weightCount = 0;

        //the first weight set will be the bias if it is a hidden node
        if (nodeType != NodeType.INPUT) {
            weights[position] = wi;
            weights[position + 1] = wf;
            weights[position + 2] = wc;
            weights[position + 3] = wo;

            weights[position + 4] = ui;
            weights[position + 5] = uf;
            weights[position + 6] = uo;

            weights[position + 7] = bi;
            weights[position + 8] = bf;
            weights[position + 9] = bc;
            weights[position + 10] = bo;

            weightCount += 11;
        }

        for (Edge edge : outputEdges) {
            weights[position + weightCount] = edge.weight;
            weightCount++;
        }

        for (RecurrentEdge recurrentEdge : outputRecurrentEdges) {
            weights[position + weightCount] = recurrentEdge.weight;
            weightCount++;
        }

        return weightCount;
    }

    /**
     * We need to override the getDeltas from RecurrentNode as
     * an LSTMNode will have 11 weights and biases as opposed to
     * just one bias.
     *
     * @param position is the index to start setting deltas in the deltas parameter
     * @param deltas is the array of deltas we're setting.
     *
     * @return the number of deltas set in the deltas parameter
     */
    public int getDeltas(int position, double[] deltas) {
        int deltaCount = 0;

        //the first delta set will be the bias if it is a hidden node
        if (nodeType != NodeType.INPUT) {
            deltas[position] = delta_wi;
            deltas[position + 1] = delta_wf;
            deltas[position + 2] = delta_wc;
            deltas[position + 3] = delta_wo;

            deltas[position + 4] = delta_ui;
            deltas[position + 5] = delta_uf;
            deltas[position + 6] = delta_uo;

            deltas[position + 7] = delta_bi;
            deltas[position + 8] = delta_bf;
            deltas[position + 9] = delta_bc;
            deltas[position + 10] = delta_bo;

            deltaCount += 11;
        }

        for (Edge edge : outputEdges) {
            deltas[position + deltaCount] = edge.weightDelta;
            deltaCount++;
        }

        for (RecurrentEdge recurrentEdge : outputRecurrentEdges) {
            deltas[position + deltaCount] = recurrentEdge.weightDelta;
            deltaCount++;
        }

        return deltaCount;
    }


    /**
     * We need to override the getDeltas from RecurrentNode as
     * an LSTMNode will have 11 weights and biases as opposed to
     * just one bias.
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
        if (nodeType != NodeType.INPUT) {
            wi = weights[position];
            wf = weights[position + 1];
            wc = weights[position + 2];
            wo = weights[position + 3];

            ui = weights[position + 4];
            uf = weights[position + 5];
            uo = weights[position + 6];

            bi = weights[position + 7];
            bf = weights[position + 8];
            bc = weights[position + 9];
            bo = weights[position + 10];

            weightCount += 11;
        }

        for (Edge edge : outputEdges) {
            edge.weight = weights[position + weightCount];
            weightCount++;
        }

        for (RecurrentEdge recurrentEdge : outputRecurrentEdges) {
            recurrentEdge.weight = weights[position + weightCount];
            weightCount++;
        }

        return weightCount;
    }

    double sigmoid(double value) {
        return 1.0 / (1.0 + Math.exp(-value));
    }

    /**
     * This propagates the postActivationValue at this LSTM node
     * to all it's output nodes.
     */
    public void propagateForward(int timeStep) {
        //TODO: You need to implement this for Programming Assignment 2 - Part 4
        //NOTE: recurrent edges need to be propagated forward from this timeStep to
        //their targetNode at timeStep + the recurrentEdge's timeSkip
        // If this is an INPUT node, just copy preActivationValue to postActivationValue.
        if (nodeType == NodeType.INPUT) {
            postActivationValue[timeStep] = preActivationValue[timeStep];
            return;
        }

        // 1) The raw input x(t) is in preActivationValue[timeStep].
        double x_t = preActivationValue[timeStep];

        // 2) If timeStep > 0, c(t-1) is ct[timeStep-1]. Otherwise, cPrev=0.
        double cPrev = (timeStep > 0) ? ct[timeStep - 1] : 0.0;

        // 3) Compute gates using c(t-1) (peephole LSTM).
        //    We do not reference h(t-1). That’s the difference from the standard LSTM.
        double affine_f = wf * x_t + uf * cPrev + bf;
        double f_t = sigmoid(affine_f);

        double affine_i = wi * x_t + ui * cPrev + bi;
        double i_t = sigmoid(affine_i);

        double affine_o = wo * x_t + uo * cPrev + bo;
        double o_t = sigmoid(affine_o);

        // 4) Candidate cell is tanh(Wc * x_t + bc).
        double affine_c = wc * x_t + bc;
        double cand_t   = Math.tanh(affine_c);

        // 5) Update the cell state: c(t) = f(t)*c(t-1) + i(t)*candidate
        double c_t = (f_t * cPrev) + (i_t * cand_t);

        // 6) Final output h(t) = o(t)*c(t)  (identity activation of that product)
        double h_t = o_t * c_t;

        // 7) Store computed values for backprop
        ft[timeStep] = f_t;
        it[timeStep] = i_t;
        ot[timeStep] = o_t;
        C[timeStep]  = cand_t;
        ct[timeStep] = c_t;
        postActivationValue[timeStep] = h_t;

        // 8) Propagate forward to feedforward edges at the same time step
        for (Edge edge : outputEdges) {
            edge.outputNode.preActivationValue[timeStep] += h_t * edge.weight;
        }

        // 9) Propagate forward to recurrent edges (timeStep + timeSkip)
        for (RecurrentEdge re : outputRecurrentEdges) {
            int nextT = timeStep + re.timeSkip;
            if (nextT < maxSequenceLength) {
                re.outputNode.preActivationValue[nextT] += h_t * re.weight;
            }
        }

    }

    /**
     * This propagates the delta back from this node
     * to its incoming edges.
     */
    public void propagateBackward(int timeStep) {
        //TODO: You need to implement this for Programming Assignment 2 - Part 4
        //note that delta[timeStep] is the delta coming in for the output (delta_ht in the slides)
        // If input node, skip
        if (nodeType == NodeType.INPUT) {
            return;
        }

        // 1) Gather forward pass values
        double i_t = it[timeStep];
        double f_t = ft[timeStep];
        double o_t = ot[timeStep];
        double cand = C[timeStep];
        double c_t = ct[timeStep];

        // 2) The error wrt h(t)
        double dh = delta[timeStep];

        // 3) Combine derivative wrt c(t)
        double dct_val = delta_ct[timeStep] + (dh * o_t);
        delta_ct[timeStep] = dct_val; // store

        double cPrev = (timeStep > 0) ? ct[timeStep - 1] : 0.0;

        // Gate partials
        double d_o = dh * c_t;
        double d_affine_o = d_o * (o_t * (1.0 - o_t));

        double d_f = dct_val * cPrev;
        double d_affine_f = d_f * (f_t * (1.0 - f_t));

        double d_i = dct_val * cand;
        double d_affine_i = d_i * (i_t * (1.0 - i_t));

        double d_cand = dct_val * i_t;
        double d_affine_c = d_cand * (1.0 - cand * cand);

        // Accumulate biases
        delta_bi += d_affine_i;
        delta_bf += d_affine_f;
        delta_bo += d_affine_o;
        delta_bc += d_affine_c;

        // Peephole weights
        double cPrevVal = cPrev;
        delta_ui += d_affine_i * cPrevVal;
        delta_uf += d_affine_f * cPrevVal;
        delta_uo += d_affine_o * cPrevVal;

        // Input weights
        double x_t = preActivationValue[timeStep];
        delta_wi += d_affine_i * x_t;
        delta_wf += d_affine_f * x_t;
        delta_wo += d_affine_o * x_t;
        delta_wc += d_affine_c * x_t;

        // partial wrt c(t-1)
        double dcPrevVal = (d_affine_f * uf) + (d_affine_i * ui) + (d_affine_o * uo);
        if (timeStep > 0) {
            delta_ct[timeStep - 1] += (dct_val * f_t) + dcPrevVal;
        }

        // partial wrt x(t)
        double dx_t = (d_affine_f * wf) + (d_affine_i * wi)
                + (d_affine_o * wo) + (d_affine_c * wc);

        // Now backprop dx_t to feedforward edges
        for (Edge e : inputEdges) {
            e.propagateBackward(timeStep, dx_t);
        }

        // And backprop dx_t to the recurrent edges
        for (RecurrentEdge re : inputRecurrentEdges) {
            re.propagateBackward(timeStep, dx_t);
        }

    }

    /**
     *  This sets the node's bias to the bias parameter and then
     *  randomly initializes each incoming edge weight by using
     *  Random.nextGaussian() / sqrt(N) where N is the number
     *  of incoming edges.
     *
     *  @param biasVal is the bias to initialize this node's bias to
     */
    public void initializeWeightsAndBiasKaiming(int fanIn, double biasVal) {
        //TODO: You need to implement this for Programming Assignment 2 - Part 4
        // Set biases: note the forget gate bias is increased by 1.0.
        bi = biasVal;
        bf = biasVal + 1.0;
        bc = biasVal;
        bo = biasVal;

        double scale = 1.0 / Math.sqrt(fanIn);
        Random rand = new Random();

        // Initialize LSTM node weights:
        wi = rand.nextGaussian() * scale;
        wf = rand.nextGaussian() * scale;
        wc = rand.nextGaussian() * scale;
        wo = rand.nextGaussian() * scale;

        ui = rand.nextGaussian() * scale;
        uf = rand.nextGaussian() * scale;
        uo = rand.nextGaussian() * scale;

        // Also initialize the weights of incoming feedforward edges.
        for (Edge e : inputEdges) {
            e.weight = rand.nextGaussian() * scale;
        }
        // And initialize the weights of incoming recurrent edges.
        for (RecurrentEdge re : inputRecurrentEdges) {
            re.weight = rand.nextGaussian() * scale;
        }


    }

    /**
     *  This sets the node's bias to the bias parameter and then
     *  randomly intializes each incoming edge weight uniformly
     *  at random (you can use Random.nextDouble()) between 
     *  +/- sqrt(6) / sqrt(fan_in + fan_out) 
     *
     *  @param biasVal is the bias to initialize this node's bias to
     */
    public void initializeWeightsAndBiasXavier(int fanIn, int fanOut, double biasVal) {
        //TODO: You need to implement this for Programming Assignment 2 - Part 2
        // Set biases: note that forget gate bias is biasVal + 1.0.
        bi = biasVal;
        bf = biasVal + 1.0;
        bc = biasVal;
        bo = biasVal;

        double limit = Math.sqrt(6.0 / (fanIn + fanOut));
        Random rand = new Random();

        // Initialize LSTM weights uniformly:
        wi = (rand.nextDouble() * 2.0 * limit) - limit;
        wf = (rand.nextDouble() * 2.0 * limit) - limit;
        wc = (rand.nextDouble() * 2.0 * limit) - limit;
        wo = (rand.nextDouble() * 2.0 * limit) - limit;

        ui = (rand.nextDouble() * 2.0 * limit) - limit;
        uf = (rand.nextDouble() * 2.0 * limit) - limit;
        uo = (rand.nextDouble() * 2.0 * limit) - limit;

        // Also initialize incoming feedforward edges.
        for (Edge e : inputEdges) {
            e.weight = (rand.nextDouble() * 2.0 * limit) - limit;
        }
        // And initialize incoming recurrent edges.
        for (RecurrentEdge re : inputRecurrentEdges) {
            re.weight = (rand.nextDouble() * 2.0 * limit) - limit;
        }

    }


    /**
     * Prints concise information about this node.
     *
     * @return The node as a short string.
     */
    public String toString() {
        return "[LSTM Node - layer: " + layer + ", number: " + number + ", type: " + nodeType + "]";
    }
}
