/**
 * Basic Neural Network class which can feed forward input as well as back propagate errors. Uses a sigmoid function
 * for the activation function and gradient descent for back propagation. Can be be set to employ either batch learning
 * or on-line/stochastic learning for back propagation.
 * @author Jason Gould
 */
class NeuralNetwork {
    private static final double MOMENTUM = .5;
    private final int numInputs;
    private final int numOutputs;
    private final int numHidden;
    private final double learningRate;
    private final boolean batch;
    private double[][] weightInputToHidden;
    private double[][] weightHiddenToOutput;
    private double[][] inputToHiddenErrorOffset;
    private double[][] hiddenToOutputErrorOffset;

    NeuralNetwork(int _numInputs, int _numOutputs, int _numHidden, double _learningRate, boolean _batch,
                         double[][] _weightInputToHidden, double[][] _weightHiddenToOutput) {
        numInputs = _numInputs;
        numOutputs = _numOutputs;
        numHidden = _numHidden;
        learningRate = _learningRate;
        batch = _batch;
        weightInputToHidden = _weightInputToHidden;
        weightHiddenToOutput = _weightHiddenToOutput;
        inputToHiddenErrorOffset = new double[numInputs+1][numHidden];
        hiddenToOutputErrorOffset = new double[numHidden+1][numOutputs];
    }

    void process(NNDataSet dataSet) {
        dataSet.reset();
        feedForward(dataSet.getInputNeuronValue(), dataSet.getHiddenNeuronRawValue(),
                    dataSet.getHiddenNeuronValue(), numHidden, weightInputToHidden);
        feedForward(dataSet.getHiddenNeuronValue(), dataSet.getOutputNeuronRawValue(),
                    dataSet.getOutputNeuronValue(), numOutputs, weightHiddenToOutput);
    }

    private void feedForward(Double[] feedValues, Double[] rawValues, Double[] resultValues,
                             int numResults, double[][] weightFeedToResult) {
        double sum;

        for(int resultIndex = 0; resultIndex < numResults; resultIndex++) {
            sum = 0;
            for(int feedIndex = 0; feedIndex < feedValues.length; feedIndex++) {
                sum += feedValues[feedIndex] * weightFeedToResult[feedIndex][resultIndex];
            }
            rawValues[resultIndex] = sum;
            resultValues[resultIndex] = activationFunction(sum);
        }
    }

    private double activationFunction(double rawValue) {
        return (1.0 / (1.0 + Math.exp(-rawValue)));
    }

    private double activationDerivative(double rawValue) {
        double denomVal = (1.0 + Math.exp(-rawValue));
        return Math.exp(-rawValue)/(denomVal * denomVal);
    }

    void backPropagate(NNDataSet dataSet) {
        double[] outputErrorDeltas = calcOutputErrorDeltas(dataSet.getOutputNeuronValue(),
                                                           dataSet.getOutputNeuronRawValue(),
                                                           dataSet.getDesiredOutputValue());
        double[] hiddenErrorDeltas = calcHiddenErrorDeltas(dataSet.getHiddenNeuronRawValue(),
                                                           outputErrorDeltas);
        calculateErrorOffsets(inputToHiddenErrorOffset, hiddenErrorDeltas, dataSet.getInputNeuronValue());
        calculateErrorOffsets(hiddenToOutputErrorOffset, outputErrorDeltas, dataSet.getHiddenNeuronValue());
    }

    private double[] calcOutputErrorDeltas(Double[] outputNeuronValue, Double[] outputNeuronRawValue,
                                           Double[] desiredOutputValue) {
        double[] errorDeltas = new double[outputNeuronValue.length];

        for(int index=0; index < errorDeltas.length; index++) {
            errorDeltas[index] =  activationDerivative(outputNeuronRawValue[index]) *
                    (desiredOutputValue[index] - outputNeuronValue[index]);
        }

        return errorDeltas;
    }

    private double[] calcHiddenErrorDeltas(Double[] hiddenNeuronRawValue,
                                           double[] outputErrorDeltas) {
        double[] errorDeltas = new double[hiddenNeuronRawValue.length];
        double weightedSum;

        for(int hiddenIndex=0; hiddenIndex < hiddenNeuronRawValue.length; hiddenIndex++) {
            weightedSum = 0;
            for(int outputIndex = 0; outputIndex < numOutputs; outputIndex++) {
                weightedSum += weightHiddenToOutput[hiddenIndex][outputIndex] * outputErrorDeltas[outputIndex];
            }

            errorDeltas[hiddenIndex] = activationDerivative(hiddenNeuronRawValue[hiddenIndex]) * weightedSum;
        }

        return errorDeltas;
    }

    private void calculateErrorOffsets(double[][] errorOffset, double[] errorDeltas, Double[] values) {

        for(int destIndex = 0; destIndex < errorDeltas.length; destIndex++) {
            for(int originIndex = 0; originIndex < values.length; originIndex++) {
                if(batch) {
                    errorOffset[originIndex][destIndex] += learningRate * values[originIndex] * errorDeltas[destIndex];
                } else {
                    errorOffset[originIndex][destIndex] = learningRate * values[originIndex] * errorDeltas[destIndex] +
                            (MOMENTUM * errorOffset[originIndex][destIndex]);
                }
            }
        }
    }

    void updateWeights() {
        for(int inputIndex = 0; inputIndex <= numInputs; inputIndex++) {
            for(int hiddenIndex = 0; hiddenIndex < numHidden; hiddenIndex++) {
                weightInputToHidden[inputIndex][hiddenIndex] += inputToHiddenErrorOffset[inputIndex][hiddenIndex];
            }
        }

        for(int hiddenIndex = 0; hiddenIndex <= numHidden; hiddenIndex++) {
            for(int outputIndex = 0; outputIndex < numOutputs; outputIndex++) {
                weightHiddenToOutput[hiddenIndex][outputIndex] += hiddenToOutputErrorOffset[hiddenIndex][outputIndex];
            }
        }

        if (batch) {
            inputToHiddenErrorOffset = new double[numInputs + 1][numHidden];
            hiddenToOutputErrorOffset = new double[numHidden + 1][numOutputs];
        }
    }
}
