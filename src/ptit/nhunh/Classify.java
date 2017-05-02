package ptit.nhunh;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;

import ca.uwo.csd.ai.nlp.kernel.CompositeKernel;
import ca.uwo.csd.ai.nlp.kernel.KernelManager;
import ca.uwo.csd.ai.nlp.kernel.LinearKernel;
import ca.uwo.csd.ai.nlp.kernel.RBFKernel;
import ca.uwo.csd.ai.nlp.libsvm.svm_model;
import ca.uwo.csd.ai.nlp.libsvm.svm_parameter;
import ca.uwo.csd.ai.nlp.libsvm.ex.Instance;
import ca.uwo.csd.ai.nlp.libsvm.ex.SVMPredictor;
import ca.uwo.csd.ai.nlp.libsvm.ex.SVMTrainer;
import utils.DataFileReader;

public class Classify {
	public void classified(String trainFile, String testFile, String outFile) throws ClassNotFoundException, IOException{
		LinearKernel(trainFile, testFile, outFile);
	}
	private void LinearKernel(String trainFile, String testFile, String outFile) throws IOException, ClassNotFoundException {
		Instance[] trainingInstances = DataFileReader.readDataFile(trainFile);
		svm_parameter param = new svm_parameter();
		KernelManager.setCustomKernel(new LinearKernel());
		System.out.println("Training started...");
		svm_model model = SVMTrainer.train(trainingInstances, param);
		System.out.println("Training completed.");

		Instance[] testingInstances = DataFileReader.readDataFile(testFile);
		double[] predictions = SVMPredictor.predict(testingInstances, model, true);
		writeOutputs(outFile, predictions);
	}

	private void writeOutputs(String outputFileName, double[] predictions) throws IOException {
		BufferedWriter writer = new BufferedWriter(new FileWriter(outputFileName));
		for (double p : predictions) {
			writer.write(String.format("%.0f\n", p));
		}
		writer.close();
	}
}
