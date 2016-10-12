package com.abevieiramota.knnann.knn;

import com.abevieiramota.knnann.knn.distance.DistanceMetric;

public class KNearestNeighbors implements NearestNeighbors {
	
	private static final KNearestNeighbors INSTANCE = new KNearestNeighbors();
	
	public static NearestNeighbors instance() {
		return INSTANCE;
	}
	
	private KNearestNeighbors() {
	}

	@Override
	public int[] nNeighbors(int n, double[][] x, double[] p, DistanceMetric distanceMetric) {
		// calcula distancias
		double[] distances = new double[x.length];
		for (int i = 0; i < x.length; i++) {
			distances[i] = distanceMetric.distance(x[i], p);
		}

		// calcula ordem dos neighbors 
		int[] kNeighborsIndex = new int[n];
		for (int i = 0; i < kNeighborsIndex.length; i++) {
			kNeighborsIndex[i] = 0;
		}
		for (int i = 0; i < kNeighborsIndex.length; i++) {
			for (int j = 0; j < distances.length; j++) {
				if (distances[kNeighborsIndex[i]] > distances[j]) {
					kNeighborsIndex[i] = j;
				}
			}
		}

		return kNeighborsIndex;
	}
	
}
