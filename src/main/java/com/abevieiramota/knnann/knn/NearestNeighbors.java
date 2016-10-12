package com.abevieiramota.knnann.knn;

import com.abevieiramota.knnann.knn.distance.DistanceMetric;

public interface NearestNeighbors {

	int[] nNeighbors(int n, double[][] x, double[] p, DistanceMetric distanceMetric);
}
