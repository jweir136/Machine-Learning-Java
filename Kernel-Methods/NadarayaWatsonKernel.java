import java.lang.*;
import java.util.*;

public class NadarayaWatsonKernel extends Model {
  private double[][] X;
  private double[] y;
  private int k;
  private double lambda;

  public NadarayaWatsonKernel(int k, double lambda) {
    this.k = k;
    this.lambda = lambda;
  }

  public void fit(double[][] X, double[] y) {
    assert X[0].length == 1 : "[-]\tError: Features must be 1 dimensional.";
    this.X = X;
    this.y = y;
  }

  public void train() {
    System.out.println("[!]\tModel Trained.");
  }

  public double[] predict(double[][] X) {
    double[] predictions = new double[X.length];
    double result;
    int counter = 0;
    ArrayList<Integer> values;
    int index;

    for (double[] x : X) {
      values = new ArrayList<>(getDistances(this.X, x).values());
      result = 0.0;

      for (int i = 0; i < this.k; i++) {
        index = values.get(i);
        result += quadKernel(x, this.X[index]) * this.y[index];
      }

      predictions[counter++] = result / this.k;
    }

    return predictions;
  }

  public double error(double[][] X, double[] y) {
    double errors = 0.0;
    double[] predictions = predict(X);

    for (int i = 0; i < y.length; i++) {
      errors += Math.pow(y[i] - predictions[i], 2);
    }

    return errors / y.length;
  }

  private double quadKernel(double[] x1, double[] x2) {
    double kernelValue = norm(subtract(x2, x1)) / this.lambda;
    if (Math.abs(kernelValue) <= 1.0) {
      return (3.0 / 4.0) * (1.0 - Math.pow(kernelValue, 2));
    }
    return 0.0;
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
