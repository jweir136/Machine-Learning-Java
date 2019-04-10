public class CrossValidation {
  private double[][] X;
  private double[] y;
  private int cv;
  private Model model;

  public CrossValidation(Model model, int cv) {
    this.model = model;
    this.cv = cv;
  }

  public void fit(double[][] X, double[] y) {
    this.X = X;
    this.y = y;
  }

  public double[] eval() {
    double[] scores = new double[this.cv];
    int index;

    for (int i = 0; i < this.cv; i++) {
      System.out.println("CV=" + i);
      double[][] trainX = new double[(int)(this.X.length * 0.9)][this.X[0].length];
      double[] trainY = new double[(int)(this.X.length * 0.9)];
      double[][] testX = new double[(int)(this.X.length * 0.1)][this.X[0].length];
      double[] testY = new double[(int)(this.X.length * 0.1)];

      int[] chosenIndices = new int[(int)(this.X.length * 0.9)];

      for (int j = 0; j < (int)(this.X.length * 0.9); j++) {
        index = (int)(Math.random() * ((this.X.length-1 - 0) + 1)) + 0;
        trainX[j] = this.X[index];
        //set the train y.
        trainY[j] = this.y[index];

        chosenIndices[j] = index;
      } 
      // fit the model.
      this.model.fit(trainX, trainY);
      this.model.train();

      scores[i] = this.model.error(trainX, trainY);
    }
    
    return scores;
  }
}
