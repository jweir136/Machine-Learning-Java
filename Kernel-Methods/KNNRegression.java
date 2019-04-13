import java.util.*;
import java.lang.*;

public class KNNRegression extends Model {
  private double[][] X;
  private double[] y;
  private int k;

  public KNNRegression(int k) {
    this.k = k;
  }

  public void fit(double[][] X, double[] y) {
    this.X = X;
    this.y = y;
  }

  public void train() {
    System.out.println("[!]\tModel Trained.");
  }

  public double[] predict(double[][] X) {
    double[] predictions = new double[X.length];
    double result;
    ArrayList<Integer> values;
    int counter = 0;

    for (double[] x : X) {
      result = 0.0;
      values = new ArrayList<Integer>(getDistances(this.X, x).values());

      for (int i = 0; i < this.k; i++) {
        result += this.y[values.get(i)];
      }

      predictions[counter++] = result / this.k;
    }

    return predictions;
  }

  public double error(double[][] X, double[] y) {
    double error = 0.0;
    double[] predictions = predict(X);

    for (int i = 0; i < X.length; i++) {
      error += Math.pow(y[i] - predictions[i], 2);
    }

    return error / X.length;
  }

  private SortedMap<Double, Integer> getDistances(double[][] X, double[] vector) {
    SortedMap<Double, Integer> distances = new TreeMap<Double, Integer>();
    double[] subtractedVector;

    for (int i = 0; i < X.length; i++) {
      subtractedVector = subtract(X[i], vector);
      distances.put(new Double(norm(subtractedVector)), new Integer(i));
    }

    return distances;
  }

  private double norm(double[] vector) {
    double result = 0.0;

    for (int i = 0; i < vector.length; i++) {
      result += Math.pow(vector[i], 2);
    }

    return Math.sqrt(result);
  }

  private double[] subtract(double[] vector1, double[] vector2) {
    double[] result = new double[vector1.length];

    for (int i = 0; i < vector1.length; i++) {
      result[i] = vector1[i] - vector2[i];
    }

    return result;
  }
}
