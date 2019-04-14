import java.util.*;
import java.lang.*;

public class BootstrapModel {
  private double[][] X;
  private double[] y;
  private Model model;
  private int numModels;

  public BootstrapModel(Model model, int numModels) {
    this.model = model;
    this.numModels = numModels;
  }

  public void fit(double[][] X, double[] y) {
    this.X = X;
    this.y = y;
  }

  public double[] predict(double[][] X) {
    double[] predictions = new double[X.length];
    double[][] trainingX;
    double[] trainingY;
    int trainSize = (int)(this.X.length * 0.9);
    double result = 0.0;
    int index;

    for (int i = 0; i < X.length; i++) {
      for (int j = 0; j < this.numModels; j++) {
        result = 0.0;
        trainingX = new double[trainSize][this.X[0].length];
        trainingY = new double[trainSize];
        for (int k = 0; k < trainSize; k++) {
          index = (int)(Math.random() * this.X.length);
          trainingX[k] = this.X[index];
          trainingY[k] = this.y[index];
        }
        this.model.fit(trainingX, trainingY);
        this.model.train();
        result += this.model.predict(X)[i];
        this.model.close();
      }
      predictions[i] = result / this.numModels;
    }
    return predictions;
  }

  public double error(double[][] X, double[] y) {
    double error = 0.0;
    double[] predictions = predict(X);

    for (int i = 0; i < predictions.length; i++) {
      error += Math.pow(y[i] - predictions[i], 2);
    }
    
    return error / predictions.length;
  }
}
