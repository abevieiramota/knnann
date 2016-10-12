package com.abevieiramota.knnann.knn;

import java.io.IOException;
import java.net.URL;
import java.nio.charset.Charset;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.ListIterator;
import java.util.Random;

import org.junit.BeforeClass;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;

import com.abevieiramota.knnann.knn.KNearestNeighbors;
import com.abevieiramota.knnann.knn.Knn;
import com.abevieiramota.knnann.knn.distance.DistanceMetric;
import com.google.common.base.Charsets;
import com.google.common.base.Splitter;
import com.google.common.collect.BiMap;
import com.google.common.collect.HashBiMap;
import com.google.common.collect.Lists;
import com.google.common.io.CharSource;
import com.google.common.io.Resources;

@RunWith(Parameterized.class)
public class TestIrisDataset {
	private static final long RANDOM_SEED = 10;

	private static final String LABEL_IRIS_VIRGINICA = "Iris-virginica";
	private static final String LABEL_IRIS_VERSICOLOR = "Iris-versicolor";
	private static final String LABEL_IRIS_SETOSA = "Iris-setosa";
	private static final Charset CHARSET_UCI_IRIS_DATA_SET = Charsets.UTF_8;
	private static final String URL_UCI_IRIS_DATA_SET = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data";
	private static final BiMap<String, Integer> MAP_IRIS_LABEL_INT = HashBiMap.create();
	private static final int N_FEATURES = 4;
	static {
		MAP_IRIS_LABEL_INT.put(LABEL_IRIS_SETOSA, 0);
		MAP_IRIS_LABEL_INT.put(LABEL_IRIS_VERSICOLOR, 1);
		MAP_IRIS_LABEL_INT.put(LABEL_IRIS_VIRGINICA, 2);
	}
	private static List<String> linhasIrisDataSet;
	@BeforeClass
	public static void beforeClass() throws IOException {
		CharSource charSource = Resources.asCharSource(new URL(URL_UCI_IRIS_DATA_SET), CHARSET_UCI_IRIS_DATA_SET);
		linhasIrisDataSet = Lists.newArrayList(charSource.readLines());
		Collections.shuffle(linhasIrisDataSet, new Random(RANDOM_SEED));
	}

	@Parameters(name = "{index}: knn({0}, {1})")
	public static Collection<Object[]> params() {
		float[] trainPcts = { 0.1f, 0.2f, 0.7f };
		int[] nsNeighbors = { 1, 2, 5, 10 };
		DistanceMetric[] distanceMetrics = { DistanceMetric.getEuclidian(), DistanceMetric.getManhattan() };
		Object[][] params = new Object[trainPcts.length * nsNeighbors.length * distanceMetrics.length][3];
		int k = 0;
		for (int l = 0; l < distanceMetrics.length; l++) {
			for (int i = 0; i < trainPcts.length; i++) {
				for (int j = 0; j < nsNeighbors.length; j++) {
					params[k][0] = trainPcts[i];
					params[k][1] = nsNeighbors[j];
					params[k][2] = distanceMetrics[l];
					k++;
				}
			}
		}
		return Arrays.asList(params);
	}

	private int nNeighbors;
	private int trainSize;
	private DistanceMetric distanceMetric;

	public TestIrisDataset(float trainPct, int nNeighbors, DistanceMetric distanceMetric) {
		this.trainSize = Math.round(trainPct * linhasIrisDataSet.size());
		this.nNeighbors = nNeighbors;
		this.distanceMetric = distanceMetric;
	}

	@Test
	public void testIrisDataset() throws IOException {

		List<String> trainXLinhas = linhasIrisDataSet.subList(0, this.trainSize);
		double[][] trainX = new double[this.trainSize][N_FEATURES];
		int[] trainY = new int[this.trainSize];
		ListIterator<String> linhasTrainIter = trainXLinhas.listIterator();
		while (linhasTrainIter.hasNext()) {
			int ix = linhasTrainIter.nextIndex();
			String linha = linhasTrainIter.next();
			if (linha.isEmpty()) {
				break;
			}
			List<String> cols = Splitter.on(',').splitToList(linha);
			for (int i = 0; i < N_FEATURES; i++) {
				trainX[ix][i] = Double.parseDouble(cols.get(i));
			}
			trainY[ix] = MAP_IRIS_LABEL_INT.get(cols.get(N_FEATURES));
		}

		List<String> testXLinhas = linhasIrisDataSet.subList(this.trainSize, linhasIrisDataSet.size());
		int testSize = linhasIrisDataSet.size() - trainSize;
		double[][] testX = new double[testSize][N_FEATURES];
		int[] testY = new int[testSize];
		ListIterator<String> linhasTestIter = testXLinhas.listIterator();
		while (linhasTestIter.hasNext()) {
			int ix = linhasTestIter.nextIndex();
			String linha = linhasTestIter.next();
			if (linha.isEmpty()) {
				break;
			}
			List<String> cols = Splitter.on(',').splitToList(linha);
			for (int i = 0; i < N_FEATURES; i++) {
				testX[ix][i] = Double.parseDouble(cols.get(i));
			}
			testY[ix] = MAP_IRIS_LABEL_INT.get(cols.get(N_FEATURES));
		}

		Knn knn = new Knn(this.nNeighbors, KNearestNeighbors.instance(), distanceMetric);
		knn.fit(trainX, trainY);

		int[] resultY = knn.predict(testX);

		Integer nMatch = 0;
		for (int i = 0; i < testY.length; i++) {
			if (testY[i] == resultY[i]) {
				nMatch++;
			}
		}

		System.out.println("[" + this.distanceMetric + "]\ttrainSize=" + this.trainSize + "\tnNeighbors="
				+ this.nNeighbors + "\tresultado=" + nMatch.doubleValue() / testY.length);
	}
}
