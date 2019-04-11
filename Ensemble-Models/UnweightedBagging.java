public class UnweightedBagging {
  private double[][] X;
  private double[] y;
  private Model[] models;

  public UnweightedBagging(Model[] models) {
    this.models = models;
  }

  public void fit(double[][] X, double[] y) {
    this.X = X;
    this.y = y;
  }

  private double sum(double[] values) {
		double sum = 0.0;
		
		for (double val : values) {
			sum += val;
		}
		
		return sum;
	}

  public void train() {
    System.out.println("[!] Models Trained!");
  }

  public double[] baggingPredict(double[][] X) {
    double[] predictions = new double[X.length];
    int counter = 0;


    return predictions;
  }

  public double error(double[][] X, double[] y) {
    double[] errors = new double[X.length];

    for (int i = 0; i < this.models.length; i++) {
      this.models[i].fit(this.X, this.y);
      this.models[i].train();
      
      errors[i] = this.models[i].error(X, y);
    }

    return sum(errors) / errors.length;
  }
}
