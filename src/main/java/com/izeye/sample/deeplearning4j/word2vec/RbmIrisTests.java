package com.izeye.sample.deeplearning4j.word2vec;

import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.IrisDataSetIterator;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.RBM;
import org.deeplearning4j.nn.layers.factory.LayerFactories;
import org.deeplearning4j.nn.params.DefaultParamInitializer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;

/**
 * Created by izeye on 16. 3. 25..
 */
public class RbmIrisTests {

	public static void main(String[] args) {
		Nd4j.MAX_SLICES_TO_PRINT = -1;
		Nd4j.MAX_ELEMENTS_PER_SLICE = -1;
		Nd4j.ENFORCE_NUMERICAL_STABILITY = true;
		
		int numRows = 4;
		int numColumns = 1;
		int numOutputs = 10;
		int numSamples = 150;
		int batchSize = 150;
		int iterations = 100;
		int seed = 123;
		int printIterations = iterations / 2;
		
		int numEpochs = 20;

		System.out.println("Loading data...");
		DataSetIterator iterator = new IrisDataSetIterator(batchSize, numSamples);
		DataSet dataSet = iterator.next();
		
		dataSet.normalizeZeroMeanZeroUnitVariance();

		System.out.println("Building model...");
		NeuralNetConfiguration configuration = new NeuralNetConfiguration.Builder()
				.regularization(true)
				.miniBatch(true)
				.layer(
						new RBM.Builder()
								.l2(1e-1)
								.l1(1e-3)
								.nIn(numRows * numColumns)
								.nOut(numOutputs)
								.activation("relu")
								.weightInit(WeightInit.RELU)
								.lossFunction(
										LossFunctions.LossFunction.RECONSTRUCTION_CROSSENTROPY)
								.k(3)
								.hiddenUnit(RBM.HiddenUnit.RECTIFIED)
								.visibleUnit(RBM.VisibleUnit.GAUSSIAN)
								.updater(Updater.ADAGRAD)
								.gradientNormalization(
										GradientNormalization.ClipL2PerLayer).build()
				)
				.seed(seed)
				.iterations(iterations)
				.learningRate(1e-3)
				.optimizationAlgo(OptimizationAlgorithm.LBFGS).build();
		
		Layer layer = LayerFactories.getFactory(configuration.getLayer()).create(configuration);
		layer.setListeners(new ScoreIterationListener(printIterations));

		System.out.println("Evaluating weights...");
		INDArray weights = layer.getParam(DefaultParamInitializer.WEIGHT_KEY);
		System.out.println("Weights: " + weights);
		
		System.out.println("Scaling the data set...");
		dataSet.scale();

		System.out.println("Training model...");
		for (int i = 0; i < numEpochs; i++) {
			System.out.println("Epoch " + i + ":");
			layer.fit(dataSet.getFeatureMatrix());
		}
	}
	
}
