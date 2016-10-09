package com.abevieiramota.knnann.distance;

class EuclidianDistance implements DistanceMetric {
	
	private static final EuclidianDistance INSTANCE = new EuclidianDistance();
	
	public static DistanceMetric instance() {
		return INSTANCE;
	}
	
	private EuclidianDistance() {
	}
	
	@Override
	public double distance(double[] x1, double[] x2) {
		if(x1.length != x2.length) {
			throw new IllegalArgumentException();
		}
		double distance = 0D;
		for(int i = 0; i < x1.length; i++) {
			distance += Math.pow((x1[i] - x2[i]), 2D);
		}
		return Math.sqrt(distance);
	}
	
	@Override
	public String toString() {
		return "Euclidian";
	}
}
