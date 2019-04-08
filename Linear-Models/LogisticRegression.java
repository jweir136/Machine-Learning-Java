import java.lang.Math;

public class LogisticRegression extends Model {
	private double[][] X;
	private double[] y;
	public double[] weights;
	private double learningRate;
	private int maxEpochs;
	private double[][] features;
	
	public LogisticRegression(int maxEpochs, double learningRate) {
		this.learningRate = learningRate;
		this.maxEpochs = maxEpochs;
	}
	
	private double dot(double[] v1, double[] v2) {
		double sum = 0.0;
		
		for (int i = 0; i < v1.length; i++) {
			sum += (v1[i] * v2[i]);
		}
		
		return sum;
	}
	
	private double sigmoid(double x) {
		return 1.0 / (1.0 + Math.exp(-x));
	}
	
	public void fit(double[][] X, double[] y) {
		this.X = X;
		this.y = y;
		this.weights = new double[this.X[0].length];
		
		for (int i = 0; i < this.X[0].length; i++) {
			this.weights[i] = 1e-10;
		}
		
		this.features = new double[this.weights.length][this.X.length];
		
		sortFeatures();
	}
	
	public double[] predict(double[][] X) {
		double[] scores = new double[X.length];
		double[] predictions = new double[X.length];
		
		for (int i = 0; i < X.length; i++) {
			scores[i] = dot(this.weights, X[i]);
			predictions[i] = sigmoid(scores[i]);
		}
		
		return predictions;
	}
	
	public double error(double[][] X, double[] y) {
		double[] preds = predict(X);
		double error = 0.0;
		
		for (int i = 0; i < X.length; i++) {
			error += preds[i] - y[i];
		}
		
		return error / X.length;
	}
	
	private void sortFeatures() {
		for (int i = 0; i < this.X[0].length; i++) {
			double[] data = new double[this.X.length];
			
			for (int j = 0; j < this.X.length; j++) {
				data[j] = this.X[j][i];
			}
			
			this.features[i] = data;
		}
	}
	
	public void train() {
		for (int i = 0; i < this.maxEpochs; i++) {
			double[] preds = predict(this.X);
			double[] errors = new double[this.X.length];
			
			for (int j = 0; j < this.X.length; j++) {
				errors[j] = this.y[j] - sigmoid(preds[j]);
			}
			
			for (int k = 0; k < this.features.length; k++) {
				double gradient = dot(this.features[k], errors);
				this.weights[k] = this.weights[k] + (this.learningRate * gradient);
			}
			
			System.out.println("[!]\tEpoch=" + i + "\tError=" + error(this.X, this.y));
		}
	}
}
