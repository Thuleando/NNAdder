/**
 * Class which holds both a data set that is used to train a neural network as well as the values achieved at both
 * the hidden and output layers when run through a neural network.
 * @author Jason Gould
 */
public class NNDataSet {
    private Double[] inputNeuronValue;
    private Double[] hiddenNeuronValue;
    private Double[] hiddenNeuronRawValue;
    private Double[] outputNeuronValue;
    private Double[] outputNeuronRawValue;
    private Double[] desiredOutputValue;
    private int numInputs;
    private int numHidden;
    private int numOutputs;

    NNDataSet(Double[] _inputNeuronValue, Double[] _desiredOutputValue,  int _numInputs, int _numOutputs, int _numHidden) {
        numInputs = _numInputs;
        numHidden = _numHidden;
        numOutputs = _numOutputs;
        inputNeuronValue = _inputNeuronValue;
        desiredOutputValue = _desiredOutputValue;
        hiddenNeuronValue = new Double[numHidden+1];
        hiddenNeuronRawValue = new Double[numHidden];
        outputNeuronValue = new Double[numOutputs];
        outputNeuronRawValue = new Double[numOutputs];

        inputNeuronValue[numInputs] = 1.0;
        hiddenNeuronValue[numHidden] = 1.0;
    }

    Double[] getInputNeuronValue() { return inputNeuronValue; }
    Double[] getHiddenNeuronValue() { return hiddenNeuronValue; }
    Double[] getHiddenNeuronRawValue() { return hiddenNeuronRawValue; }
    Double[] getOutputNeuronValue() { return outputNeuronValue; }
    Double[] getOutputNeuronRawValue() { return outputNeuronRawValue; }
    Double[] getDesiredOutputValue() { return desiredOutputValue; }
    void reset() {
        hiddenNeuronValue = new Double[numHidden+1];
        hiddenNeuronValue[numHidden] = 1.0;
        hiddenNeuronRawValue = new Double[numHidden];
        outputNeuronValue = new Double[numOutputs];
        outputNeuronRawValue = new Double[numOutputs];
    }

    @Override
    public String toString() {
        StringBuilder outputString = new StringBuilder();
        outputString.append("Inputs:  \t[");
        for(int index = 0; index < numInputs; index++) {
            outputString.append(inputNeuronValue[index].toString());
            outputString.append(", ");
        }
        outputString.replace(outputString.length()-2, outputString.length(), "");
        outputString.append("]\n");
        outputString.append("Outputs:\t[");
        for(Double output: outputNeuronValue) {
            outputString.append(String.format("%1$.2f, ", output));
        }
        outputString.replace(outputString.length()-2, outputString.length(), "");
        outputString.append("]\n");
        outputString.append("Desired:\t[");
        for(Double desired: desiredOutputValue) {
            outputString.append(desired.toString());
            outputString.append(", ");
        }
        outputString.replace(outputString.length()-2, outputString.length(), "");
        outputString.append("]\n");

        return outputString.toString();
    }
}
