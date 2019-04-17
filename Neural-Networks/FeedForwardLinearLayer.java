public class FeedForwardLinearLayer extends Model {
  private double[][] X;
  private double[] y;
  private double learningRate, weight;
  private int maxEpochs;

  public FeedForwardLinearLayer(double learningRate, int maxEpochs) {
    this.learningRate = learningRate;
    this.maxEpochs = maxEpochs;
    this.weight = Math.random();
  }

  public void fit(double[][] X, double[] y) {
    this.X = X;
    this.y = y;
  }

  public void train() {
    double[] pastWeights = new double[this.maxEpochs];
    pastWeights[0] = this.weight;

    for (int i = 0; i < this.maxEpochs; i++) {
      this.weight -= this.learningRate * deriv();

      if ((this.weight - pastWeights[pastWeights.length-2]) < 1e-3) {
        System.out.println("[+]\tConverged at epoch=" + i);
        break;
      }

      pastWeights[i+1] = this.weight;

      System.out.println("[!]\tEpoch=" + i + "\tMSE=" + error(this.X, this.y));
    }
  }

  public double[] predict(double[][] X) {
    double[] predictions = new double[X.length];

    for (int i = 0; i < X.length; i++) {
      predictions[i] = this.weight * X[i][0];
    }

    return predictions;
  }

  public double error(double[][] X, double[] y) {
    double[] predictions = predict(X);
    double error = 0.0;

    for (int i = 0; i < X.length; i++) {
      error += Math.pow(y[i] - predictions[i], 2);
    }

    return error / X.length;
  }

  public void close() {
    this.weight = Math.random();
  }

  private double deriv() {
    double[] predictions = predict(this.X);
    double error = 0.0;

    for (int i = 0; i < this.X.length; i++) {
      error += -(this.y[i] - predictions[i]);
    }

    return (2.0 * error) / this.y.length;
  }
}
