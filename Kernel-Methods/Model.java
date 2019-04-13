public abstract class Model {
	abstract void fit(double[][] X, double[] y);
	abstract double error(double[][] X, double[] y);
	abstract double[] predict(double[][] X);
	abstract void train();
}
