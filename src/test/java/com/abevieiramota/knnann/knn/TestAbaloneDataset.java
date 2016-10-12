package com.abevieiramota.knnann.knn;

import java.io.IOException;
import java.net.URL;
import java.nio.charset.Charset;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.ListIterator;
import java.util.Map;
import java.util.Random;
import java.util.stream.Stream;

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
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import com.google.common.io.CharSource;
import com.google.common.io.Resources;

@RunWith(Parameterized.class)
public class TestAbaloneDataset {
	private static final long RANDOM_SEED = 10;

	private static final Charset CHARSET_UCI_ABALONE_DATA_SET = Charsets.UTF_8;
	private static final String URL_UCI_ABALONE_DATA_SET = "https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data";
	private static final int N_FEATURES = 8;
	private static List<String> linhasAbaloneDataSet;
	private static final Map<String, Double[]> MAP_SEX_TO_INT = Maps.newHashMap();
	static {
		MAP_SEX_TO_INT.put("M", new Double[] { 0D, 0D, 1D });
		MAP_SEX_TO_INT.put("F", new Double[] { 0D, 1D, 0D });
		MAP_SEX_TO_INT.put("I", new Double[] { 1D, 0D, 0D });
	}
	private static final double[] FEATURE_WEIGHTS = { 4D, 4D, 4D, 0.8D, 0.9D, 2.3D, 0.7D, 3.1D, 2.8D, 2.3D };

	@BeforeClass
	public static void beforeClass() throws IOException {
		CharSource charSource = Resources.asCharSource(new URL(URL_UCI_ABALONE_DATA_SET), CHARSET_UCI_ABALONE_DATA_SET);
		linhasAbaloneDataSet = Lists.newArrayList(charSource.readLines());
		Collections.shuffle(linhasAbaloneDataSet, new Random(RANDOM_SEED));
	}

	@Parameters(name = "{index}: knn({0}, {1})")
	public static Collection<Object[]> params() {
		float[] trainPcts = { 0.2f, 0.7f };
		int[] nsNeighbors = { 10, 50, 100 };
		DistanceMetric[] distanceMetrics = { DistanceMetric.getEuclidian(), DistanceMetric.getManhattan(),
				DistanceMetric.getAb(FEATURE_WEIGHTS) };
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

	public TestAbaloneDataset(float trainPct, int nNeighbors, DistanceMetric distanceMetric) {
		this.trainSize = Math.round(trainPct * linhasAbaloneDataSet.size());
		this.nNeighbors = nNeighbors;
		this.distanceMetric = distanceMetric;
	}

	/**
	 * nome desse método ficou sexy 
	 */
	private static double[] getOneHotSex(String sex) {

		return Stream.of(MAP_SEX_TO_INT.get(sex)).mapToDouble(Double::doubleValue).toArray();
	}

	// gambiarras, gambiarras everywhere
	@Test
	public void testIrisDataset() throws IOException {

		List<String> trainXLinhas = linhasAbaloneDataSet.subList(0, this.trainSize);
		double[][] trainX = new double[this.trainSize][N_FEATURES + 2];
		int[] trainY = new int[this.trainSize];
		ListIterator<String> linhasTrainIter = trainXLinhas.listIterator();
		while (linhasTrainIter.hasNext()) {
			int ix = linhasTrainIter.nextIndex();
			String linha = linhasTrainIter.next();
			if (linha.isEmpty()) {
				break;
			}
			List<String> cols = Splitter.on(',').splitToList(linha);
			System.arraycopy(getOneHotSex(cols.get(0)), 0, trainX[ix], 0, 3);
			for (int i = 3; i < N_FEATURES + 1; i++) {
				trainX[ix][i] = Double.parseDouble(cols.get(i - 2));
			}
			trainY[ix] = Integer.parseInt(cols.get(N_FEATURES));
		}

		List<String> testXLinhas = linhasAbaloneDataSet.subList(this.trainSize, linhasAbaloneDataSet.size());
		int testSize = linhasAbaloneDataSet.size() - trainSize;
		double[][] testX = new double[testSize][N_FEATURES + 2];
		int[] testY = new int[testSize];
		ListIterator<String> linhasTestIter = testXLinhas.listIterator();
		while (linhasTestIter.hasNext()) {
			int ix = linhasTestIter.nextIndex();
			String linha = linhasTestIter.next();
			if (linha.isEmpty()) {
				break;
			}
			List<String> cols = Splitter.on(',').splitToList(linha);
			System.arraycopy(getOneHotSex(cols.get(0)), 0, testX[ix], 0, 3);
			for (int i = 3; i < N_FEATURES + 1; i++) {
				testX[ix][i] = Double.parseDouble(cols.get(i - 2));
			}
			testY[ix] = Integer.parseInt(cols.get(N_FEATURES));
		}

		Knn knn = new Knn(this.nNeighbors, KNearestNeighbors.instance(), this.distanceMetric);
		knn.fit(trainX, trainY);

		int[] resultY = knn.predict(testX);

		Integer nMatch = 0;
		for (int i = 0; i < testY.length; i++) {
			if (testY[i] == resultY[i]) {
				nMatch++;
			}
		}

		System.out.println("[" + this.distanceMetric + "]\ttrainSize=" + this.trainSize + "\tnNeighbors="
				+ this.nNeighbors + "\tresultado=" + String.format("%.2f", nMatch.doubleValue() / testY.length));
	}
}
