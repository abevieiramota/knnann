package com.abevieiramota.knnann.knn.distance;

public interface DistanceMetric {

	public static DistanceMetric getEuclidian() {
		return EuclidianDistance.instance();
	}

	public static DistanceMetric getManhattan() {
		return ManhattanDistance.instance();
	}

	public static DistanceMetric getAb() {
		return AbeDistance.instance();
	};
	
	public static DistanceMetric getAb(double[] weights) {
		return AbeDistance.newInstance(weights);
	}

	double distance(double[] x1, double[] x2);

	default double distance(double[] x1, double[] x2, double[] weights) {
		return distance(x1, x2);
	};
}
