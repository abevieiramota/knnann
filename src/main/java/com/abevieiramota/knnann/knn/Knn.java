package com.abevieiramota.knnann.knn;

import java.util.Arrays;
import java.util.Collections;
import java.util.Map;
import java.util.function.Function;
import java.util.stream.Collectors;

import com.abevieiramota.knnann.knn.distance.DistanceMetric;

// brute force search
public class Knn {

	private int nNeighbors;
	private double[][] trainX;
	private int[] trainY;
	private NearestNeighbors nearest;
	private DistanceMetric distanceMetric;

	public Knn(int nNeighbors, NearestNeighbors nearest, DistanceMetric distanceMetric) {
		this.nNeighbors = nNeighbors;
		this.nearest = nearest;
		this.distanceMetric = distanceMetric;
	}

	public void fit(double[][] x, int[] y) {
		if (x.length < this.nNeighbors) {
			throw new IllegalArgumentException();
		}
		this.trainX = x;
		this.trainY = y;
	}

	public int[] predict(double[][] ds) {
		int[] labels = new int[ds.length];
		for (int j = 0; j < ds.length; j++) {
			labels[j] = predict(ds[j]);
		}

		return labels;
	}

	private int predict(double[] p) {
		int[] kNeighborsIndex = this.nearest.nNeighbors(this.nNeighbors, this.trainX, p, this.distanceMetric);

		// conta o label para os k neighbors
		// TODO: ponto de alteração para weighted knn
		Map<Integer, Long> groupByCount = Arrays.stream(kNeighborsIndex).map(i -> this.trainY[i]).boxed()
				.collect(Collectors.groupingBy(Function.identity(), Collectors.counting()));

		return Collections.max(groupByCount.entrySet(), (e1, e2) -> e1.getValue().compareTo(e2.getValue())).getKey();
	}
}
