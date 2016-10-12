package com.abevieiramota.knnann.ann;

import static org.junit.Assert.assertEquals;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import com.abevieiramota.knnann.knn.distance.DistanceMetric;

@RunWith(JUnit4.class)
public class TestAnn {
	
	@Test
	public void testMalha() {
		Node n1 = new Node(new double[]{0D, 0D});
		Node n2 = new Node(new double[]{0D, 1D});
		Node n3 = new Node(new double[]{0D, 2D});
		
		n1.connectTo(n2);
		n1.connectTo(n3);
		
		n1.addEnergy(10D);
		
		n1.propagate(1, DistanceMetric.getEuclidian());
		
		assertEquals(5D, n2.getEnergy(), 0D);
		assertEquals(3.3D, n3.getEnergy(), 0.04D);
	}
	
	@Test
	public void testLoop() {
		Node n1 = new Node(new double[]{0D, 4D});
		Node n2 = new Node(new double[]{0D, 3D});
		
		n1.connectTo(n2);
		n2.connectTo(n1);
		
		n1.addEnergy(12D);
		
		n1.propagate(2, DistanceMetric.getEuclidian());
		
		assertEquals(6D, n2.getEnergy(), 0D);
		assertEquals(15D, n1.getEnergy(), 0D);
	}
}
