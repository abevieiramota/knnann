package com.abevieiramota.knnann.distance;

class AbeDistance implements DistanceMetric {

	private static final AbeDistance INSTANCE = new AbeDistance();

	public static DistanceMetric instance() {
		return INSTANCE;
	}
	public static DistanceMetric newInstance(double[] weights) {
		return new AbeDistance(weights);
	}

	private double[] weights;

	private AbeDistance() {
	}
	private AbeDistance(double[] weights) {
		this.weights = weights;
	}

	@Override
	public double distance(double[] x1, double[] x2) {
		if (x1.length != x2.length) {
			throw new IllegalArgumentException();
		}
		if(this.weights != null) {
			return _distanceWithWeights(x1, x2);
		} else {
			return _distance(x1, x2);
		}
	}
	
	private double _distance(double[] x1, double[] x2) {
		double euclidian = 0D;
		double manhattan = 0D;
		for (int i = 0; i < x1.length; i++) {
			euclidian += Math.pow((x1[i] - x2[i]), 2D);
			manhattan += Math.abs(x1[i] - x2[i]);
		}

		return Math.sqrt(euclidian * manhattan);
	}
	
	private double _distanceWithWeights(double[] x1, double[] x2) {
		double euclidian = 0D;
		double manhattan = 0D;
		for (int i = 0; i < x1.length; i++) {
			euclidian += Math.pow((x1[i] - x2[i]), 2D) * this.weights[i];
			manhattan += Math.abs(x1[i] - x2[i]) * this.weights[i];
		}

		return Math.sqrt(euclidian * manhattan);
	}

	@Override
	public String toString() {
		return "Aqui é a distância do AB!";
	}

}
