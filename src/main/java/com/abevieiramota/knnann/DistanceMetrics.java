package com.abevieiramota.knnann;

public final class DistanceMetrics {
	
	private static final DistanceMetric euclidian = new EuclidianDistance();

	public static DistanceMetric getEuclidian() {
		return euclidian;
	}
}
