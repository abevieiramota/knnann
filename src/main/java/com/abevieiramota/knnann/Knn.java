package com.abevieiramota.knnann;

import java.util.Arrays;
import java.util.Collections;
import java.util.Map;
import java.util.function.Function;
import java.util.stream.Collectors;

// brute force search
public class Knn {

	private int nNeighbors;
	private DistanceMetric distanceMetric;
	private double[][] trainX;
	private int[] trainY;

	public Knn(int nNeighbors, DistanceMetric distanceMetric) {
		this.nNeighbors = nNeighbors;
		this.distanceMetric = distanceMetric;
	}

	public void fit(double[][] x, int[] y) {
		if(x.length < this.nNeighbors) {
			throw new IllegalArgumentException();
		}
		this.trainX = x;
		this.trainY = y;
	}

	public int[] predict(double[][] ds) {
		int[] labels = new int[ds.length];
		// calcula distancias
		for (int j = 0; j < ds.length; j++) {
			labels[j] = predict(ds[j]);
		}

		return labels;
	}

	private int predict(double[] p) {
		// calcula distancias
		double[] distances = new double[trainX.length];
		for (int i = 0; i < trainX.length; i++) {
			distances[i] = this.distanceMetric.distance(this.trainX[i], p);
		}

		// calcula ordem dos neighbors 
		int[] kNeighborsIndex = new int[this.nNeighbors];
		for(int i = 0; i < kNeighborsIndex.length; i++) {
			kNeighborsIndex[i] = 0;
		}
		for (int i = 0; i < kNeighborsIndex.length; i++) {
			for (int j = 0; j < distances.length; j++) {
				if (distances[kNeighborsIndex[i]] > distances[j]) {
					kNeighborsIndex[i] = j;
				}
			}
		}
		// calcula label dos k neighbors
		int[] kNeighborsLabels = new int[this.nNeighbors];
		for (int i = 0; i < this.nNeighbors; i++) {
			kNeighborsLabels[i] = this.trainY[kNeighborsIndex[i]];
		}

		// conta o label para os k neighbors
		Map<Integer, Long> groupByCount = Arrays.stream(kNeighborsLabels).boxed()
				.collect(Collectors.groupingBy(Function.identity(), Collectors.counting()));

		return Collections.max(groupByCount.entrySet(), (e1, e2) -> e1.getValue().compareTo(e2.getValue())).getKey();
	}
}
