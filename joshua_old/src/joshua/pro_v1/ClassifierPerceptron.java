package joshua.pro_v1;

import java.util.Vector;

public class ClassifierPerceptron implements ClassifierInterface 
{
	@Override
	public double[] runClassifier(Vector<String> samples, double[] initialLambda, int featDim) 
	{
		System.out.println("------- Average-perceptron training starts ------");
		
		int sampleSize = samples.size();
		double score = 0;  //model score
		double label;
		double[] lambda = new double[featDim+1];  //in ZMERT lambda[0] is not used
		double[] sum_lambda = new double[featDim+1];
		String[] featVal;
		
		//in ZMERT lambda[0] is not used
		for( int i=1; i<=featDim; i++ )
		{
			sum_lambda[i] = 0;
			lambda[i] = initialLambda[i];
		}
		
		System.out.print("Perceptron iteration ");
		int numError = 0;
		//int numPosSamp = 0;
		
		for( int it=0; it<maxIter; it++ )
		{
			System.out.print(it + " ");
			numError = 0;
			//numPosSamp = 0;
			
			for( int s=0; s<sampleSize; s++ )
			{
				featVal = samples.get(s).split("\\s+");
				
				//only consider positive samples
				//if( featVal[featDim].equals("1") )
				//{
					//numPosSamp++;
					score = 0;
					for( int d=0; d<featDim; d++ )  //inner product
					{
						//System.out.printf("%.2f ", Double.parseDouble(featVal[d]));
						score += Double.parseDouble(featVal[d]) * lambda[d+1];  //in ZMERT lambda[0] is not used
					}
					//System.out.println();
					
					label = Double.parseDouble(featVal[featDim]);
					score *= label;  //the last element is class label(+1/-1)
					
					if( score<=bias ) //incorrect classification
					{
						numError++;
						
						for( int d=0; d<featDim; d++ )
						{
							lambda[d+1] += learningRate*label * Double.parseDouble(featVal[d]);
							sum_lambda[d+1] += learningRate*lambda[d+1];
						}
					}
				//}//if( featVal[featDim].equals("1") )
			}
			
			//System.out.printf("(%.2f%%) ",numError*100.0/numPosSamp);
			
			if( numError == 0 )
				break;
		}
		
		System.out.println("\n------- Average-perceptron training ends ------");
		
		for( int i=1; i<=featDim; i++ )
				sum_lambda[i] /= maxIter;
		
		return sum_lambda;
	}
	
	@Override
	/* for avg_perceptron:
	 * param[0] = maximum number of iterations
	 * param[1] = learning rate (step size)
	 * param[2] = bias (usually set to 0)
	 */
	public void setClassifierParam( String[] param )
	{
		if( param == null )
			System.out.println("WARNING: no parameters specified for perceptron classifier, using default settings.");
		else
		{
			maxIter = Integer.parseInt(param[0]);
			learningRate = Double.parseDouble(param[1]);
			bias = Double.parseDouble(param[2]);
		}
	}
	
	int maxIter = 20;
	double learningRate = 0.5;
	double bias = 0.0;
}
