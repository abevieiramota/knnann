package com.abevieiramota.knnann;

import com.abevieiramota.knnann.distance.DistanceMetric;

public interface NearestNeighbors {

	int[] nNeighbors(int n, double[][] x, double[] p, DistanceMetric distanceMetric);
}
