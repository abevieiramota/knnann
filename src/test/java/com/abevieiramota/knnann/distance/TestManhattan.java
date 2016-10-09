package com.abevieiramota.knnann.distance;

import static org.junit.Assert.assertEquals;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public class TestManhattan {
	
	@Test
	public void testDistanciaEntrePontosIguais() {
		double[] x = { 0D, 1D };
		assertEquals(0D, DistanceMetric.getManhattan().distance(x, x), 0D);
	}

	@Test
	public void testDistanciaEntrePontos1() {
		double[] x1 = { 0D, 3D };
		double[] x2 = { 4D, 0D };
		assertEquals(7D, DistanceMetric.getManhattan().distance(x1, x2), 0D);
	}

	@Test(expected = IllegalArgumentException.class)
	public void testDimensoesIguais() {
		double[] x1 = { 0D, 3D, 6D };
		double[] x2 = { 4D, 0D };
		DistanceMetric.getManhattan().distance(x1, x2);
	}
}
