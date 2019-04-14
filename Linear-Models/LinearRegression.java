import java.lang.Math;

public class LinearRegression extends Model {
	private double[][] X;
	private double[] y;
	private int maxEpochs;
	private double learningRate;
	public double[] coefs;
	public double intercept;
	
	public LinearRegression(int maxEpochs, double learningRate) {
		this.maxEpochs = maxEpochs;
		this.learningRate = learningRate;
	}
	
	public void fit(double[][] X, double[] y) {
		this.X = X;
		this.y = y;
		
		this.intercept = 1e-5;
		this.coefs = new double[this.X[0].length];
		
		for (int i = 0; i < this.X[0].length; i++) {
			this.coefs[i] = 1e-10;
		}
	}
	
	public double error(double[][] X, double[] y) {
		double[] errors = new double[X.length];
		
		for (int i = 0; i < X.length; i++) {
			errors[i] = (predict(X)[i] - y[i])*(predict(X)[i] - y[i]);
		}
		
		return sum(errors) / errors.length;
	}
	
	public void train() {
		double oldIntercept = 0.0;
		
		for (int i = 0; i < this.maxEpochs; i++) {
			if (hasConverged(oldIntercept, this.intercept)) {
				break;
			}
			
			this.intercept = this.intercept - (this.learningRate * derivIntercept());
			
			for (int j = 0; j < this.X[0].length; j++) {
				this.coefs[j] = this.coefs[j] - (this.learningRate * derivCoefs()[j]);
			}
						
			System.out.println("[!]\tEpoch=" + i + "\tMSE=" + error(this.X, this.y));
		}
	}
	
	private double sum(double[] values) {
		double sum = 0.0;
		
		for (double val : values) {
			sum += val;
		}
		
		return sum;
	}
	
	private double dot(double[] v1, double[] v2) {
		double result = 0.0;
		
		for (int i = 0; i < v1.length; i++) {
			result += v1[i] * v2[i];
		}
		
		return result;
	}
	
	public double[] predict(double[][] X) {
		double[] predictions = new double[X.length];
		
		for (int i = 0; i < X.length; i++) {
			predictions[i] = dot(X[i], this.coefs) + this.intercept;
		}
		
		return predictions;
	}
	
	private double derivIntercept() {
		double[] errors = new double[this.X.length];
		
		for (int i = 0; i < this.X.length; i++) {
			errors[i] = predict(this.X)[i] - this.y[i];
		}
		
		return sum(errors) / this.X.length;
	}
	
	private double[] derivCoefs() {
		double[] derivs = new double[this.X[0].length];
		
		for (int i = 0; i < this.X[0].length; i++) {
			double[] errors = new double[this.X.length];
			
			for (int j = 0; j < this.X.length; j++) {
				errors[j] = (predict(this.X)[j] - this.y[j]) * this.X[j][i];
			}
			
			derivs[i] = sum(errors) / this.X.length;
		}
		
		return derivs;
	}
	
	private boolean hasConverged(double old, double current) {
		if ((current - old) < 1e-10) {
			return true;
		}
		return false;
	}
	
	public void close() {
		for (int i = 0; i < this.coefs.length; i++) {
			this.coefs[i] = 1e-10;
		}
		this.intercept = 1e-10;
	}
}
