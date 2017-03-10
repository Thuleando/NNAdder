/**
 * Created by Jason on 3/1/2017.
 */
public class NueralNetwork {
    private final int numInputs;
    private final int numOutputs;
    private final int numHidden;
    private double[][] weightInputToHidden;
    private double[][] weightHiddenToOutput;
    private double[][] inputToHiddenErrorOffset;
    private double[][] hiddenToOutputErrorOffset;

    public NueralNetwork(int _numInputs, int _numOutputs, int _numHidden,
                         double[][] _weightInputToHidden,
                         double[][] _weightHiddenToOutput) {
        numInputs = _numInputs;
        numOutputs = _numOutputs;
        numHidden = _numHidden;
        weightInputToHidden = _weightInputToHidden;
        weightHiddenToOutput = _weightHiddenToOutput;
        inputToHiddenErrorOffset = new double[numInputs][numHidden];
        hiddenToOutputErrorOffset = new double[numHidden][numOutputs];
    }

    public void process(NNDataSet dataSet) {
        double[] inputNeuronValue = dataSet.getInputNeuronValue();
        double[] hiddenNeuronValue =  dataSet.getHiddenNeuronValue();
        double[] outputNeuronValue = dataSet.getOutputNeuronValue();
        feedForward(inputNeuronValue, hiddenNeuronValue);
        feedForward(hiddenNeuronValue, outputNeuronValue);
        backPropogate(dataSet);
    }

    private void feedForward(double[] feedValues, double[] resultValues) {
        //multiply each feed value by the weight and sum for each result node
        //apply the activation function to the previous sum and assign to the
        //result array
    }

    private double activationFunction() {
        return 0;
    }

    private void backPropogate(NNDataSet dataSet) {
        double[] inputNeuronValue = dataSet.getInputNeuronValue();
        double[] hiddenNeuronValue =  dataSet.getHiddenNeuronValue();
        double[] outputNeuronValue = dataSet.getOutputNeuronValue();
        //double[] hiddenToOutputError = calculateError(dataSet);
       // calcWeightDeltas(outputNeuronValue, hiddenNeuronValue);
       // calcWeightDeltas(hiddenNeuronValue, inputNeuronValue);
    }

    private void calcWeightDeltas(double[] somevals, double[] othervals) {

    }

    private double[] calculateError(NNDataSet dataSet) {
        double[] outputNueronValue = dataSet.getOutputNeuronValue();
        double[] desiredOutputValue = dataSet.getDesiredOutputValue();
        return new double[1];
    }

    public void updateWeights(double[] inputDeltas, double[] hiddenDeltas) {
        for(int inputNeuron = 0; inputNeuron < numInputs; ) {
            for(int hiddenNeuron = 0; hiddenNeuron < numHidden; hiddenNeuron++) {

            }
        }

        for(int hiddenNeuron = 0; hiddenNeuron < numHidden; hiddenNeuron++) {
            for(int outputNeuron = 0; outputNeuron < numOutputs; outputNeuron++) {

            }
        }

        inputToHiddenErrorOffset = new double[numInputs][numHidden];
        hiddenToOutputErrorOffset = new double[numHidden][numOutputs];
    }
}
