import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.pmml.consumer.NeuralNetwork;
import weka.classifiers.trees.*;
import weka.core.Instances;
import weka.core.converters.CSVLoader;
import java.io.*;
import weka.filters.Filter;
import weka.filters.supervised.attribute.Discretize;
import weka.filters.unsupervised.attribute.NumericToNominal;

import javax.swing.*;
import java.util.ArrayList;
import java.util.Random;

public class WekaRun {
    public static BufferedReader readDataFile(String filename) {
        BufferedReader inputReader = null;

        try {
            inputReader = new BufferedReader(new FileReader(filename));
        } catch (FileNotFoundException ex) {
            System.err.println("File not found: " + filename);
        }

        return inputReader;
    }
    public static Instances make_discrete(Instances inp) {
        Instances discretedata=null;
        try {
            Discretize  filter = new Discretize();
            filter.setInputFormat(inp);
            // apply filter
            discretedata = Filter.useFilter(inp, filter);
        } catch(Exception e)
        {}
        return discretedata;
    }
    public static Instances make_nominal_class(Instances inp) {
        Instances nominaldata=null;
        try {
            int cindex = inp.numAttributes() - 1;
            NumericToNominal convert= new NumericToNominal();
            String[] options= new String[2];
            options[0]="-R";
            options[1]=String.valueOf(cindex);  //range of variables to make numeric
            convert.setOptions(options);
            convert.setInputFormat(inp);

            nominaldata=Filter.useFilter(inp, convert);
        } catch(Exception e)
        {}
        return nominaldata;
    }
    public static Instances load_csv(String filename) {
        Instances data=null;
        try {
            BufferedReader datafile = readDataFile(filename);
            CSVLoader loader = new CSVLoader();
            loader.setSource(new File(filename));
            data = loader.getDataSet();
            data.setClassIndex(data.numAttributes() - 1);
        } catch(Exception e)
        {}
        return data;
    }
    public static void main(String[] args) throws Exception {
//        String inputFile = "data/iris.csv";
//        //Load data
//        Instances data = load_csv(inputFile);
//        //call classifiers after appropriate filters
//        //evaluate using crossvalidation
//        Classifier cls = new J48();
//        Evaluation eval = new Evaluation(data);
//        Random rand = new Random(1);  // using seed = 1
//        int folds = 10;
//        eval.crossValidateModel(cls, data, folds, rand);
//        System.out.println(eval.toClassDetailsString());
//        doHomeWork7_1();
        doHomeWork7_2();
        doHomeWork7_3();
    }

    private static void doHomeWork7_3(){
        ArrayList<Double> f1_scores = new ArrayList<Double>();
        ArrayList<String> hiddenLayers = new ArrayList<String>(){};
        ArrayList<Double> learningRates = new ArrayList<>();
        ArrayList<String> files = getFiles();

        hiddenLayers.add("1");
        hiddenLayers.add("2");

        learningRates.add(0.01);
        learningRates.add(0.1);
        learningRates.add(0.2);

        try {
            System.out.println("7.3) solution: ");
            for(String file: files){
                Instances data = load_csv(file);
                f1_scores = get_f1_score_NeuralNetwork(data, hiddenLayers, learningRates);
                System.out.println(file);
                System.out.println(hiddenLayers);
                System.out.println(learningRates);
                System.out.println(f1_scores);
            }
        }
        catch (Exception ex){
            ex.printStackTrace();
        }
    }

    private static ArrayList<Double> get_f1_score_NeuralNetwork(Instances data, ArrayList<String> hiddenLayers, ArrayList<Double> learningRates) {
        ArrayList<Double> f1_scores = new ArrayList<Double>();
        for(String hiddenLayer: hiddenLayers){
            for(double learningRate: learningRates){
                MultilayerPerceptron neuralNetwork = new MultilayerPerceptron();
                neuralNetwork.setLearningRate(learningRate);
                neuralNetwork.setHiddenLayers(hiddenLayer);

                Evaluation eval = null;
                try {
                    eval = new Evaluation(data);
                    Random rand = new Random(1);  // using seed = 1
                    int folds = 10;
                    eval.crossValidateModel(neuralNetwork, data, folds, rand);
//                    System.out.println(eval.toClassDetailsString());
                    f1_scores.add(eval.weightedFMeasure());
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }
        }
        return f1_scores;
    }

    private static void doHomeWork7_2() throws Exception {
        ArrayList<Double> f1_scores = new ArrayList<Double>();
        ArrayList<Float> confidenceFactors = new ArrayList<>();
        ArrayList<String> files = getFiles();


        for (double i = 0.1; i <= 0.5; i += 0.1) {
            confidenceFactors.add((float) i);
        }

        System.out.println("7.2 Solution)");
        for(String file: files){
            //Load data
            Instances data = load_csv(file);
            f1_scores = get_f1_score_J48(data, confidenceFactors);
            System.out.println(file);
            System.out.println(confidenceFactors);
            System.out.println(f1_scores);
        }
    }

    private static ArrayList<String> getFiles() {
        ArrayList<String> files = new ArrayList<>();
        files.add("data/iris.csv");
        files.add("data/wines.csv");

        return files;
    }

    private static ArrayList<Double> get_f1_score_J48(Instances data, ArrayList<Float> confidenceFactors) throws Exception{

        ArrayList<Double> f1_scores = new ArrayList<>();
        for (float cf : confidenceFactors) {

            //call classifiers after appropriate filters
            //evaluate using crossvalidation
            J48 cls = new J48();
            cls.setConfidenceFactor(cf);

            Evaluation eval = new Evaluation(data);
            Random rand = new Random(1);  // using seed = 1
            int folds = 10;
            eval.crossValidateModel(cls, data, folds, rand);
//            System.out.println(eval.toClassDetailsString());

            double f1_score = eval.weightedFMeasure();
            f1_scores.add(f1_score);
        }

        return f1_scores;
    }

    private static void doHomeWork7_1() {

    }

}
