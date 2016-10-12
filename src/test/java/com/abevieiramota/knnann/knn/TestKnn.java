package com.abevieiramota.knnann.knn;

import static org.junit.Assert.assertEquals;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import com.abevieiramota.knnann.knn.KNearestNeighbors;
import com.abevieiramota.knnann.knn.Knn;
import com.abevieiramota.knnann.knn.distance.DistanceMetric;

@RunWith(JUnit4.class)
// TODO: parameterized
public class TestKnn {

	/**
	 * http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html#sklearn.neighbors.KNeighborsClassifier
	 * >>> X = [[0], [1], [2], [3]]
	 * >>> y = [0, 0, 1, 1]
	 * >>> from sklearn.neighbors import KNeighborsClassifier
	 * >>> neigh = KNeighborsClassifier(n_neighbors=3)
	 * >>> neigh.fit(X, y) 
	 * KNeighborsClassifier(...)
	 * >>> print(neigh.predict([[1.1]]))
	 * [0]
	 * >>> print(neigh.predict_proba([[0.9]]))
	 * [[ 0.66666667  0.33333333]]
	 */
	@Test
	public void testInicial1() {
		Knn knn = new Knn(3, KNearestNeighbors.instance(), DistanceMetric.getEuclidian());
		double[][] X = { { 0D }, { 1D }, { 2D }, { 3D } };
		int[] y = { 0, 0, 1, 1 };

		knn.fit(X, y);

		assertEquals(0, knn.predict(X)[0]);
	}

	@Test
	public void testInicial2() {
		Knn knn = new Knn(2, KNearestNeighbors.instance(), DistanceMetric.getEuclidian());
		double[][] X = { { 0D }, { 1D }, { 2D }, { 3D } };
		int[] y = { 0, 0, 1, 1 };

		knn.fit(X, y);

		double[][] x = { { -1D }, { -2D } };

		assertEquals(0, knn.predict(x)[0]);
	}

	@Test
	public void testInicial3() {
		Knn knn = new Knn(2, KNearestNeighbors.instance(), DistanceMetric.getEuclidian());
		double[][] X = { { 0D }, { 1D }, { 2D }, { 3D } };
		int[] y = { 0, 0, 1, 1 };

		knn.fit(X, y);

		double[][] x = { { 50D }, { 200D } };

		assertEquals(1, knn.predict(x)[0]);
	}

	@Test(expected = IllegalArgumentException.class)
	public void testNNeighborsGTTrainSize() {
		Knn knn = new Knn(10, KNearestNeighbors.instance(), DistanceMetric.getEuclidian());
		double[][] X = { { 0D }, { 1D }, { 2D }, { 3D } };
		int[] y = { 0, 0, 1, 1 };

		knn.fit(X, y);

		knn.predict(X);
	}

}
