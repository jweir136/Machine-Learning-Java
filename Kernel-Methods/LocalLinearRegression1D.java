import java.util.*;
import java.lang.*;

public class LocalLinearRegression1D extends Model {
  private double[][] X;
  private double[] y;
  public double alpha, beta;
  private int epochs, k;
  private double learningRate, lambda;

  public LocalLinearRegression1D(int k, int epochs, double lambda, double learningRate) {
    this.k = k;
    this.epochs = epochs;
    this.lambda = lambda;
    this.learningRate = learningRate;
  }

  public void fit(double[][] X, double[] y) {
    this.X = X;
    this.y = y;
    this.alpha = 1e-10;
    this.beta = 1e-10;
  }

  public void train() {
    ;
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

  private double[] subtract(double[] vector1, double[] vector2) {
    double[] result = new double[vector1.length];

    for (int i = 0; i < vector1.length; i++) {
      result[i] = vector1[i] - vector2[i];
    }

    return result;
  }

  private double norm(double[] vector) {
    double result = 0.0;

    for (int i = 0; i < vector.length; i++) {
      result += Math.pow(vector[i], 2);
    }

    return Math.sqrt(result);
  }

  public double deriv(double[] x0, double[][] X, double[] y) {
    double result = 0.0;

    for (int i = 0; i < X.length; i++) {
      result += (y[i] - this.alpha * x0[0] - this.beta * x0[0] * X[i][0]) - 2 * (y[i] - this.alpha * x0[0] - this.beta * x0[0] * X[i][0]) * (quadKernel(x0[0], X[i][0]));
    }

    return result / X.length;
  }

  public double[] predict(double[][] X) {
    double[] predictions = new double[X.length];
    int index;
    int counter = 0;
    double[][] chosenX = new double[this.k][1];
    double[] chosenY = new double[this.k];
    ArrayList<Integer> values;

    for (double[] x : X) {
      values = new ArrayList<>(getDistances(X, x).values());
      for (int i = 0; i < this.k; i++) {
        index = values.get(i);
        chosenX[i] = this.X[index];
        chosenY[i] = this.y[index];
      }
      train(chosenX, chosenY, x);
      predictions[counter++] = this.alpha * x[0] + this.beta * x[0] * x[0];
      this.alpha = 1e-10;
      this.beta = 1e-10;
      chosenX = new double[this.k][1];
      chosenY = new double[this.k];
    }

    return predictions;
  }

  public double error(double[][] X, double[] y) {
    double errors = 0.0;

    return errors;
  }

  private double quadKernel(double x1, double x2) {
    double kernelValue = Math.abs(x2 - x1) / this.lambda;
    if (Math.abs(kernelValue) <= 1.0) {
      return (3.0 / 4.0) * (1.0 - Math.pow(kernelValue, 2));
    }
    return 0.0;
  }

  private boolean hasConverged(double last, double current) {
    if (last - current <= 1e-5) {
      return true;
    }
    return false;
  }

  private void train(double[][] X, double[] y, double[] x0) {
    double lastAlpha = this.alpha;
    double lastBeta = this.beta;

    for (int i = 0; i < this.epochs; i++) {
      if (i > 5 && hasConverged(lastAlpha, this.alpha) == false) {
        this.alpha -= this.learningRate * deriv(x0, X, y);
      } else if (i <= 5) {
        this.alpha -= this.learningRate * deriv(x0, X, y);
      }
      if (i > 5 && hasConverged(lastBeta, this.beta) == false) {
        this.beta -= this.learningRate * deriv(x0, X, y);
      } else if (i <= 5) {
        this.beta -= this.learningRate * deriv(x0, X, y);
      }
      if (i > 5 && hasConverged(lastAlpha, this.alpha) && hasConverged(lastBeta, this.beta)) {
        break;
      }
      lastAlpha = this.alpha;
      lastBeta = this.beta;

      System.out.println("Epoch=" + i + "\tAlpha=" + this.alpha);
    }
  }
}
