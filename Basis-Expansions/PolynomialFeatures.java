import java.lang.Math;

public class PolynomialFeatures extends BasisExpansions {
  private double[][] X;
  private double[] y;
  private double[][] transX;
  private int degree;

  public PolynomialFeatures(int degree) {
    this.degree = degree;
  }

  public void fit(double[][] X, double[] y) {
    this.X = X;
    this.y = y;
    this.transX = new double[this.X.length][this.X[0].length * this.degree];
  }

  public double[][] transform(double[][] X) {
    for (int i = 0; i < this.X.length; i++) {
      double[] newX = new double[this.X[0].length * this.degree];
      int counter = 0;
      for (int j = 1; j < this.degree+1; j++) {
        for (int k = 0; k < this.X[0].length; k++) {
          newX[counter] = Math.pow(this.X[i][k], j);
          counter++;
        }
      }
      this.transX[i] = newX;
    }
    return this.transX;
  }
}
