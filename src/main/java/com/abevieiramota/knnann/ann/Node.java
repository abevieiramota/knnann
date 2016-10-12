package com.abevieiramota.knnann.ann;

import java.util.Collection;

import com.abevieiramota.knnann.knn.distance.DistanceMetric;
import com.google.common.collect.Lists;

public class Node {

	private double energy;
	private Collection<Node> neighbors;
	private double[] coords;

	public Node(double[] coords) {
		this.coords = coords;
		this.energy = 0D;
		this.neighbors = Lists.newLinkedList();
	}

	public void connectTo(Node n2) {
		this.neighbors.add(n2);
	}

	public void addEnergy(double d) {
		this.energy += d;
	}

	public void propagate(int maxNodes, DistanceMetric distanceMetric) {
		if (maxNodes > 0) {
			for (Node neighbor : this.neighbors) {
				double distance = distanceMetric.distance(this.coords, neighbor.coords) + 1;
				neighbor.addEnergy(this.energy / distance);
				neighbor.propagate(maxNodes - 1, distanceMetric);
			}
		}
	}

	public double getEnergy() {
		return this.energy;
	}

}
