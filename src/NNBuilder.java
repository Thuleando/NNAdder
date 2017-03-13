import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.Random;

/**
 * Builds and trains a Neural Network on a provided data set.
 * @author Jason Gould
 */
public class NNBuilder {
    private static final double INVALID_RESULT = -1;
    private static final boolean BATCH = false;
    private static final HashSet<String> LABELS;
    private static final String REGEX = ",";
    private static final double DESIRED_ACCURACY = 0.9999;
    private static final int MAX_ITERATIONS = 100000;
    private static final double PORTION_OF_DATASET_FOR_GEN_VAL_SETS = 0.1;
    private static final boolean SHOW_VALIDATION_SET_RESULTS = false;
    private boolean splitDataSets;
    private String inputFileName;
    private int lineNumber;
    private int numInput;
    private int numHidden;
    private int numOutput;
    private double learningRate;
    private NNDataSet[] dataSets;
    private NNDataSet[] trainingDataSets;
    private NNDataSet[] generalizationDataSets;
    private NNDataSet[] validationDataSets;
    private ArrayList<NNDataSet> setsNotBeingEvaluatedCorrectly;
    private NeuralNetwork network;
    private double accuracy;
    private int iterations;

    static{
        LABELS = new HashSet<>();
        LABELS.add("NUM_INPUT");
        LABELS.add("NUM_OUTPUT");
        LABELS.add("NUM_HIDDEN");
        LABELS.add("LEARNING_RATE");
        LABELS.add("DATA_SET");
        LABELS.add("INPUT");
        LABELS.add("DESIRED_OUTPUT");
    }


    private NNBuilder(String _inputFileName) {
        inputFileName = _inputFileName;
        setsNotBeingEvaluatedCorrectly = new ArrayList<>();
    }

    public static void main(String[] args) {
        assert(args.length > 0);

        NNBuilder program = new NNBuilder(args[0]);
        program.init(args);
        program.run();
        //program.saveData();
    }

    private void init(String[] args) {
        try(BufferedReader inputFile = new BufferedReader(new FileReader(inputFileName))) {
            loadData(inputFile);
        } catch(IOException ex) {
            System.out.println(ex.getMessage());
            ex.printStackTrace();
            System.exit(1);
        }

        splitDataSets = false;
        if(args.length >= 2) {
            splitDataSets = Boolean.parseBoolean(args[1]);
        }
        assignDataSets(splitDataSets);
        //adjust the number of weights to account for the addition of the bias node
        double[][] weightInputToHidden = genRandomWeights(numInput+1, numHidden);
        double[][] weightHiddenToOutput = genRandomWeights(numHidden+1, numOutput);
        network = new NeuralNetwork(numInput, numOutput, numHidden, learningRate, BATCH,
                                    weightInputToHidden, weightHiddenToOutput);
    }

    private void loadData(BufferedReader inputFile) throws IOException {
        lineNumber = 0;
        boolean numInputRead = false;
        boolean numOutputRead = false;
        boolean numHiddenRead = false;
        boolean learningRateRead = false;
        //inputRead and desiredOutputRead initialized to true so that the first instance of
        //DATA_SET encountered will not reject and throw an exception
        boolean inputRead = true;
        boolean desiredOutputRead = true;
        ArrayList<Double> inputValues = new ArrayList<>();
        ArrayList<Double> desiredOutputValues = new ArrayList<>();
        ArrayList<NNDataSet> loadedDataSets = new ArrayList<>();

        String input = getNextLine(inputFile);

        while(input != null) {
            switch(input) {
                case "NUM_INPUT":
                    input = handleNumInputRead(inputFile);
                    numInputRead = true;
                    break;
                case "NUM_OUTPUT":
                    input = handleNumOutputRead(inputFile);
                    numOutputRead = true;
                    break;
                case "NUM_HIDDEN":
                    input = handleNumHiddenRead(inputFile);
                    numHiddenRead = true;
                    break;
                case "LEARNING_RATE":
                    input = handleLearningRateRead(inputFile);
                    learningRateRead = true;
                    break;
                case "DATA_SET":
                    input = handleDataSetRead(inputRead, desiredOutputRead, numHiddenRead,
                                              numInputRead, numOutputRead, learningRateRead, inputFile);
                    //Reset the flags for input and desired output values to represent the new data set read state
                    inputRead = false;
                    desiredOutputRead = false;
                    break;
                case "INPUT":
                    input = handleInputRead(inputRead, inputFile, inputValues);
                    inputRead = true;
                    if(desiredOutputRead) {
                        loadedDataSets.add(createDataSet(inputValues, desiredOutputValues));
                    }
                    break;
                case "DESIRED_OUTPUT":
                    input = handleDesiredOutputRead(desiredOutputRead, desiredOutputValues, inputFile);
                    desiredOutputRead = true;
                    if(inputRead) {
                        loadedDataSets.add(createDataSet(inputValues, desiredOutputValues));
                    }
                    break;
                default:
                    input = getNextLine(inputFile);
            }
        }
        assert (loadedDataSets.size() > 0);
        dataSets = new NNDataSet[loadedDataSets.size()];
        dataSets = loadedDataSets.toArray(dataSets);
    }

    private String getNextLine(BufferedReader file) throws IOException{
        lineNumber++;
        String nextLine = file.readLine();
        if(nextLine != null) {
            nextLine = nextLine.trim();
        }
        return nextLine;
    }

    private String handleNumInputRead(BufferedReader inputFile) throws IOException {
        String input = getNextLine(inputFile);
        numInput = Integer.parseInt(input);
        return getNextLine(inputFile);
    }

    private String handleNumOutputRead(BufferedReader inputFile) throws IOException {
        String input = getNextLine(inputFile);
        numOutput = Integer.parseInt(input);
        return getNextLine(inputFile);
    }

    private String handleNumHiddenRead(BufferedReader inputFile) throws IOException {
        String input = getNextLine(inputFile);
        numHidden = Integer.parseInt(input);
        return getNextLine(inputFile);
    }

    private String handleLearningRateRead(BufferedReader inputFile) throws IOException {
        String input = getNextLine(inputFile);
        learningRate = Double.parseDouble(input);
        return getNextLine(inputFile);
    }

    private String handleDataSetRead(boolean inputRead, boolean desiredOutputRead,
                                     boolean numHiddenRead, boolean numInputRead,
                                     boolean numOutputRead, boolean learningRateRead,
                                     BufferedReader inputFile) throws IOException {
        if (inputRead && desiredOutputRead) {
            if (!numHiddenRead || !numInputRead || !numOutputRead || !learningRateRead) {
                throw new IOException("Error: Malformed Data file. Must declare NUM_INPUT, NUM_OUTPUT," +
                        " NUM_HIDDEN, and LEARNING_RATE before a DATA_SET may be declared");
            }
        } else {
            String missingData;
            if(!inputRead && !desiredOutputRead) {
                missingData = "input data and desired output data";
            } else if (!inputRead) {
                missingData = "input data";
            } else {
                missingData = "desired output data";
            }

            throw new IOException("Error: Malformed Data file. New DataSet encountered on line " +
                    lineNumber + " before " + missingData + " was loaded for the previous DataSet.");
        }

        return getNextLine(inputFile);
    }

    private String handleInputRead(boolean inputRead, BufferedReader inputFile,
                                   ArrayList<Double> inputValues) throws IOException{
        if(inputRead) {
            throw new IOException("Error: Malformed Data File. Second input specified for a " +
                    "single DataSet on line " + lineNumber);
        }

        String[] splitInputValues;
        inputValues.clear();
        String input = getNextLine(inputFile);

        while(input != null && !LABELS.contains(input)) {
            splitInputValues = input.split(REGEX);
            for(String value: splitInputValues) {
                inputValues.add(Double.parseDouble(value));
            }
            input = getNextLine(inputFile);
        }

        return input;
    }

    private String handleDesiredOutputRead(boolean desiredOutputRead, ArrayList<Double> desiredOutputValues,
                                           BufferedReader inputFile) throws IOException {
        if(desiredOutputRead) {
            throw new IOException("Error: Malformed Data File. Second desired output specified for a " +
                    "single DataSet on line " + lineNumber);
        }

        desiredOutputValues.clear();
        String[] splitOutputValues;
        String value;
        String input = getNextLine(inputFile);

        while(input != null && !LABELS.contains(input)) {
            splitOutputValues = input.split(REGEX);
            for(String splitValue: splitOutputValues) {
                if (!(value = splitValue.trim()).isEmpty()) {
                    desiredOutputValues.add(Double.parseDouble(value));
                }
            }
            input = getNextLine(inputFile);
        }

        return input;
    }

    private NNDataSet createDataSet(ArrayList<Double> inputValues, ArrayList<Double> desiredOutputValues) {
        Double[] inputArray = new Double[numInput+1];
        Double[] desiredOutputArray = new Double[numOutput];
        return new NNDataSet(inputValues.toArray(inputArray),
                desiredOutputValues.toArray(desiredOutputArray), numInput,
                numOutput, numHidden);
    }

    private void assignDataSets(boolean splitDataSets) {
        if (splitDataSets && dataSets.length < 3) {
            System.out.println("Must supply a minimum of 3 data sets in order to use the " +
                    "training/generalization/validation data set split feature. Running using a single DataSet");
            splitDataSets = false;
        }

        if(splitDataSets) {
            int smallSubset = (int)(dataSets.length * PORTION_OF_DATASET_FOR_GEN_VAL_SETS);
            int numGenSets = smallSubset > 0 ? smallSubset : 1;
            int numValSets = smallSubset > 0 ? smallSubset : 1;
            int numTrainSets = dataSets.length - (numGenSets + numValSets);
            NNDataSet[] allSets = dataSets.clone();

            generalizationDataSets = extractRandomSets(allSets, numGenSets);
            validationDataSets = extractRandomSets(allSets, numValSets);
            trainingDataSets = new NNDataSet[numTrainSets];
            int trainingIndex = 0;
            for(NNDataSet set: allSets) {
                if(set != null) {
                    trainingDataSets[trainingIndex] = set;
                    trainingIndex++;
                }
            }
            //Check to make sure we filled the training data set array
            assert (trainingDataSets[numTrainSets-1] != null);
        } else {
            trainingDataSets = dataSets;
            generalizationDataSets = dataSets;
            validationDataSets = dataSets;
        }
    }

    private NNDataSet[] extractRandomSets(NNDataSet[] allSets, int numNeeded) {
        Random randomGenerator = new Random();
        NNDataSet[] newSets = new NNDataSet[numNeeded];
        int setsAcquired = 0;

        while(setsAcquired != numNeeded) {
            int possibleSet = randomGenerator.nextInt(allSets.length);
            if(allSets[possibleSet] != null) {
                newSets[setsAcquired] = allSets[possibleSet];
                allSets[possibleSet] = null;
                setsAcquired++;
            }
        }

        return newSets;
    }

    private double[][] genRandomWeights(int nodesInFirstLayer, int nodesInNextLayer) {
        Random randomGenerator = new Random();
        double[][] randomWeights = new double[nodesInFirstLayer][nodesInNextLayer];
        for(int firstIndex = 0; firstIndex < nodesInFirstLayer; firstIndex++) {
            for(int nextIndex = 0; nextIndex < nodesInNextLayer; nextIndex++) {
                randomWeights[firstIndex][nextIndex] = randomGenerator.nextDouble() -0.4999;
            }
        }

        return randomWeights;
    }

    private void run() {
        iterations = 0;
        accuracy = calcAccuracy(generalizationDataSets);
        outputStats();

        while(accuracy < DESIRED_ACCURACY && iterations < MAX_ITERATIONS) {
            for(NNDataSet dataSet: trainingDataSets) {
                network.process(dataSet);
                network.backPropagate(dataSet);
                if(!BATCH) {network.updateWeights();}
            }
            if(BATCH) {network.updateWeights();}
            accuracy = calcAccuracy(generalizationDataSets);
            iterations++;
            outputStats();
        }
        accuracy = calcAccuracy(validationDataSets);
        outputFinalResults();
    }

    private double calcAccuracy(NNDataSet[] setsToEval) {
        setsNotBeingEvaluatedCorrectly.clear();
        double totalCorrectOutputs =  0;
        boolean setOutputAccurate;

        for(NNDataSet dataSet: setsToEval) {
            setOutputAccurate = true;
            network.process(dataSet);
            Double[] outputValues = dataSet.getOutputNeuronValue();
            Double[] desiredValues = dataSet.getDesiredOutputValue();
            for(int valueIndex = 0; valueIndex < outputValues.length; valueIndex++) {
                if (clamp(outputValues[valueIndex]) == desiredValues[valueIndex]) {
                    totalCorrectOutputs++;
                } else {
                    setOutputAccurate = false;
                }
            }

            if(!setOutputAccurate) {
                setsNotBeingEvaluatedCorrectly.add(dataSet);
            }
        }
        return totalCorrectOutputs / (double)(setsToEval.length * numOutput);
    }

    private double clamp(double value) {
        double result;
        if(value < .1) {
            result = 0;
        } else if (value > .90) {
            result = 1;
        } else {
            result = INVALID_RESULT;
        }

        return result;
    }

    private void outputStats() {
        System.out.printf("Iteration(%1$d): %2$.2f %%\n", iterations, accuracy*100);
    }

    private void outputFinalResults() {
        System.out.printf("\nFinal Result:\nTotal Iterations:\t%1$d\nResulting Accuracy:\t%2$.2f %%\n",
                iterations, accuracy*100);

        System.out.println("\nSets the Neural Network Failed to evaluate correctly");
        if(setsNotBeingEvaluatedCorrectly.isEmpty()) {
            System.out.println("None");
        } else {
            for (NNDataSet set : setsNotBeingEvaluatedCorrectly) {
                System.out.println(set);
            }
        }

        if(SHOW_VALIDATION_SET_RESULTS) {
            System.out.println("\nAll Validation Set Results:");
            for (NNDataSet set : validationDataSets) {
                System.out.println(set);
            }
        }
    }

    private void saveData() {

    }
}
