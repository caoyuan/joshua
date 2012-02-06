package joshua.pro;

import java.io.IOException;
import java.util.Vector;

import joshua.pro.classifier.maxent.edu.stanford.nlp.classify.ColumnDataClassifier;

public class ClassifierMaxEnt implements ClassifierInterface 
{
	@Override
	public double[] runClassifier(Vector<String> samples, double[] initialLambda, int featDim) 
	{
		System.out.println("--------- Start MaxEnt training ----------");
		
		double[] lambda = new double[featDim+1];
		//propFilePath = "/home/yuan/Desktop/stanford-classifier-2011-09-16/src/examples/iris2007.prop";
				
		try
		{
			lambda = ColumnDataClassifier.run(propFilePath, samples, featDim);
		}
		catch(IOException e)
		{
			e.printStackTrace();
		}
		
		System.out.println("--------- End MaxEnt training ----------");
		
		/*
		try 
		{
			Thread.sleep(20000);
		} 
		catch(InterruptedException e) 
		{
		} */
		
		return lambda;
	}

	@Override
	/* for Stanford MaxEnt tool:
	 * param[0] = prop file path
	 */
	public void setClassifierParam(String[] param) 
	{
		if( param== null )
		{
			System.out.println("ERROR: must provide parameters for Stanford Max-Entropy classifier!");
			System.exit(10);
		}
		else
			propFilePath = param[0];
	}

	String propFilePath;
}
