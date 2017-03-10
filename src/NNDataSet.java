/**
 * Created by Jason on 3/6/2017.
 */
public class NNDataSet {
    private double[] inputNeuronValue;
    private double[] hiddenNeuronValue;
    private double[] outputNeuronValue;
    private double[] desiredOutputValue;
    private int numHidden;
    private int numOutputs;

    public NNDataSet(double[] _inputNueronValue, double[] _desiredOutputValue,  int _numOutputs, int _numHidden) {
        inputNeuronValue = _inputNueronValue;
        desiredOutputValue = _desiredOutputValue;
        numHidden = _numHidden;
        numOutputs = _numOutputs;
        hiddenNeuronValue = new double[numHidden];
        outputNeuronValue = new double[numOutputs];
    }

    public double[] getInputNeuronValue() { return inputNeuronValue; }
    public double[] getHiddenNeuronValue() { return hiddenNeuronValue; }
    public double[] getOutputNeuronValue() { return outputNeuronValue; }
    public double[] getDesiredOutputValue() { return desiredOutputValue; }
    public void reset() {
        hiddenNeuronValue = new double[numHidden];
        outputNeuronValue = new double[numOutputs];
    }

}
