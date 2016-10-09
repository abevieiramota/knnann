package com.abevieiramota.knnann.distance;

class ManhattanDistance implements DistanceMetric {
	
	private static final ManhattanDistance INSTANCE = new ManhattanDistance();
	public static DistanceMetric instance() {
		return INSTANCE;
	}
	private ManhattanDistance() {
	}

	@Override
	public double distance(double[] x1, double[] x2) {
		if(x1.length != x2.length) {
			throw new IllegalArgumentException();
		}
		double distance = 0D;
		for(int i = 0; i < x1.length; i++) {
			distance += Math.abs(x1[i] - x2[i]);
		}
		return distance;
	}
	
	@Override
	public String toString() {
		return "Manhattan";
	}

}
