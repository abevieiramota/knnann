package com.abevieiramota.knnann;

import static org.junit.Assert.assertEquals;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public class TestEuclidian {

	@Test
	public void testDistanciaEntrePontosIguais() {
		DistanceMetric e = DistanceMetrics.getEuclidian();
		double[] x = { 0D, 1D };
		assertEquals(0D, e.distance(x, x), 0D);
	}
	
	@Test
	public void testDistanciaEntrePontos1() {
		DistanceMetric e = DistanceMetrics.getEuclidian();
		double[] x1 = { 0D, 3D };
		double[] x2 = { 4D, 0D };
		assertEquals(5D, e.distance(x1, x2), 0D);
	}
	
	@Test(expected=IllegalArgumentException.class)
	public void testDimensoesIguais() {
		DistanceMetric e = DistanceMetrics.getEuclidian();
		double[] x1 = { 0D, 3D, 6D };
		double[] x2 = { 4D, 0D };
		e.distance(x1, x2);
	}
}
