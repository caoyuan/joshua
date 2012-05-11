package joshua.pro;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.TreeMap;
import java.util.Vector;

import joshua.zmert.*;

import edu.stanford.nlp.classify.ColumnDataClassifier;

//this class implements the PRO optimization method
public class Optimizer 
{
	public Optimizer( int _j, long _seed, int _sentNum, Vector<String> _output, double[] _initialLambda, 
					  HashMap<String,String>[] _feat_hash, HashMap<String,String>[] _stats_hash, 
					  double _finalScore, EvaluationMetric _evalMetric, int _Tau, int _Xi, double _metricDiff, 
					  double[] _normalizationOptions, String _optAlgorithm )
	{
		j	=	_j;
		sentNum	=	_sentNum;
		output	=	_output;
		initialLambda	=	_initialLambda;
		//finalLambda	=	_finalLambda;
		finalScore	=	_finalScore;
		feat_hash 	=	_feat_hash;
		stats_hash	=	_stats_hash;
		paramDim	=	initialLambda.length-1; //because in ZMERT lambda array is given length paramNum+1
		evalMetric	=	_evalMetric;
		Tau	=	_Tau;
		Xi	=	_Xi;
		metricDiff = _metricDiff;
		normalizationOptions	=	_normalizationOptions;
		randgen = new Random(_seed);
		optAlgorithm = _optAlgorithm;
	}
	
	public double[] run_Optimizer()
	{
		//sampling from all candidates
		Vector<String> allSamples = process_Params();
		
		//do classification
		if( optAlgorithm.equals("avg_percep") )
		{
			System.out.println("------- Using average perceptron for classification -------");
			int maxIter = 30;
			finalLambda = run_Perceptron( allSamples, initialLambda, paramDim, maxIter );
		}
		else if( optAlgorithm.equals("stanford_maxent") )
		{
			System.out.println("------- Using Stanford MaxEnt for classification -------");
			finalLambda = run_maxent(allSamples);
		}
		else if( optAlgorithm.equals("megam") )
		{
			System.out.println("------- Using MegaM for classification -------");
			finalLambda = run_MegaM(allSamples);
		}
		else if( optAlgorithm.equals("svm") )
		{
			System.out.println("------- Using SVM for classification -------");
			finalLambda = run_svm(allSamples);
		}
	
		normalizeLambda(finalLambda);
		
		String result = "Final lambda is: ";
		for( int i=1; i<=paramDim; i++ )  //in ZMERT finalLambda[0] is not used
			result += finalLambda[i]+" ";
		
		output.add(result);
		
		return finalLambda;
	}
	
	public Vector<String> process_Params()
	{
		Vector<String> allSamples = new Vector<String>();  //to save all sampled pairs
		
		//sampling
		Vector<String> sampleVec = new Vector<String>();  //use String to make sparse representation easy
		for( int i=0; i<sentNum; i++ )
		{
			sampleVec = Sampler(i);
			allSamples.addAll(sampleVec);
		}
		
		return allSamples;
	}
	
	private Vector<String> Sampler( int sentId )
	{
		int candCount = stats_hash[sentId].size();
		Vector<String> sampleVec = new Vector<String>();
		HashMap<String, Double> candScore = new HashMap<String, Double>();  //metric(e.g BLEU) score of all candidates
		
		//extract all candidates to a string array to save time in computing BLEU score
		String[] cands = new String[candCount];
		Set<String> candSet = stats_hash[sentId].keySet();
		HashMap<Integer, String> candMap = new HashMap<Integer, String>();
		
		int candId = 0;
		for(Iterator it=candSet.iterator(); it.hasNext(); )
		{
			cands[candId] = it.next().toString();
			candMap.put(candId, cands[candId]); //map an integer to each candidate
			candId++;
		}
		candScore = compute_Score(sentId, cands);  //compute BLEU for each candidate
		
		//start to sample
		double scoreDiff;
		double probAccept;
		boolean accept;
		HashMap<String, Double> acceptedPair = new HashMap<String, Double>();
		
		if( Tau < candCount*(candCount-1) ) //otherwise no need to sample
		{
			int j1, j2;
			for( int i=0; i<Tau; i++ )
			{
				//here the case in which the same pair is sampled more than once is allowed
				//otherwise if Tau is almost the same as candCount^2, it might take a lot of time to find
				//Tau distinct pairs
				j1 = randgen.nextInt(candCount);
				j2 = randgen.nextInt(candCount);
				while(j1==j2)
					j2 = randgen.nextInt(candCount);
				
				//accept or not?
				scoreDiff = Math.abs(candScore.get(candMap.get(j1)) - candScore.get(candMap.get(j2)));
				probAccept = Alpha(scoreDiff);
				
				accept = randgen.nextDouble() <= probAccept ? true : false;  
				 
				if(accept)
					acceptedPair.put(j1+" "+j2, scoreDiff);
			}
		}
		else
		{	
			for( int i=0; i<candCount; i++ )
			{
				for ( int j=0; j<candCount; j++ )
				{
					if(j!=i)
					{	
						//accept or not?
						scoreDiff = Math.abs( candScore.get(candMap.get(i)) - candScore.get(candMap.get(j)) );
						probAccept = Alpha(scoreDiff);
						
						accept = randgen.nextDouble() <= probAccept ? true : false;
						 
						if(accept)
							acceptedPair.put(i+" "+j, scoreDiff);
					}
				}
			}
		}
		
		//System.out.println("Tau="+Tau+"\nAll possible pair number: "+candCount*(candCount-1));
		//System.out.println("Number of accepted pairs after random selection: "+acceptedNum);
		
		//sort sampled pairs according to "scoreDiff"
		ValueComparator comp = new ValueComparator(acceptedPair);
		TreeMap<String, Double> acceptedPairSort = new TreeMap<String, Double>(comp);
		acceptedPairSort.putAll(acceptedPair);
		
		int topCount = 0;
		int label;
		String[] pair_str;
		String[] feat_str_j1, feat_str_j2;
		String j1Cand, j2Cand;
		String featDiff, neg_featDiff;
		HashSet<String> added = new HashSet<String>();  //to avoid symmetric duplicate
		
		for( String key :  acceptedPairSort.keySet() )
		{
			if( topCount == Xi )
				break;
			
			pair_str = key.split("\\s+");
			//System.out.println(pair_str[0]+" "+pair_str[1]+" "+acceptedPair.get(key));
			
			if( ! added.contains(key) )
			{
				j1Cand = candMap.get(Integer.parseInt(pair_str[0]));
				j2Cand = candMap.get(Integer.parseInt(pair_str[1]));
				label = (candScore.get(j1Cand) - candScore.get(j2Cand)) > 0 ? 1 : -1;
				
				feat_str_j1 = feat_hash[sentId].get(j1Cand).split("\\s+");
				feat_str_j2 = feat_hash[sentId].get(j2Cand).split("\\s+");
	
				featDiff = "";
				neg_featDiff = "";
				for( int i=0; i<feat_str_j1.length; i++ )
				{
					featDiff += ( Double.parseDouble(feat_str_j1[i]) - Double.parseDouble(feat_str_j2[i]) ) + " ";
					neg_featDiff += ( Double.parseDouble(feat_str_j2[i]) - Double.parseDouble(feat_str_j1[i]) ) + " ";
				}
				featDiff += label;
				neg_featDiff += -label;
				
				//System.out.println(featDiff);
				
				//System.out.println(featDiff + " | " + candScore.get(j1Cand) + " " + candScore.get(j2Cand));
				//System.out.println(neg_featDiff + " | " + acceptedPair.get(key));
				
				sampleVec.add(featDiff);
				sampleVec.add(neg_featDiff);
	
				//both (j1,j2) and (j2,j1) have been added to training set
				added.add(key);
				added.add(pair_str[1]+" "+pair_str[0]);
				
				topCount++;
			}
		}
		
		//System.out.println("Selected top "+topCount+ "pairs for training");
		
		return sampleVec;
	}
	
	private double Alpha(double x)
	{
		return x < metricDiff ? 0 : 1;  //default implementation of the paper's method
								  //other functions possible
	}
	
	private HashMap<String, Double> compute_Score( int sentId, String[] cands )
	{
		HashMap<String, Double> candScore = new HashMap<String, Double>();
		String statString;
		String[] statVal_str;
		int[] statVal = new int[evalMetric.get_suffStatsCount()];
		
		//for all candidates
		for( int i=0; i<cands.length; i++ )
		{
			statString = stats_hash[sentId].get(cands[i]);
			statVal_str = statString.split("\\s+");
			
			for( int j=0; j<evalMetric.get_suffStatsCount(); j++ )
				statVal[j] = Integer.parseInt(statVal_str[j]);
			
			candScore.put(cands[i], evalMetric.score(statVal));
		}
		
		return candScore;
	}
	
	private double[] run_Perceptron( Vector<String> samples, double[] initialLambda, int paramDim, int maxIter )
	{
		int sampleSize = samples.size();
		double score = 0;  //model score
		double label;
		double[] lambda = new double[paramDim+1];  //in ZMERT lambda[0] is not used
		double[] sum_lambda = new double[paramDim+1];
		String[] featVal;
		
		System.out.println("Total training samples(positive+negative): "+sampleSize);
		
		//in ZMERT lambda[0] is not used
		for( int i=1; i<=paramDim; i++ )
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
				//if( featVal[paramDim].equals("1") )
				//{
					//numPosSamp++;
					score = 0;
					for( int d=0; d<paramDim; d++ )  //inner product
					{
						//System.out.printf("%.2f ", Double.parseDouble(featVal[d]));
						score += Double.parseDouble(featVal[d]) * lambda[d+1];  //in ZMERT lambda[0] is not used
					}
					//System.out.println();
					
					label = Double.parseDouble(featVal[paramDim]);
					score *= label;  //the last element is class label(+1/-1)
					
					if( score<=0 ) //incorrect classification
					{
						numError++;
						
						for( int d=0; d<paramDim; d++ )
						{
							lambda[d+1] += 0.5*label * Double.parseDouble(featVal[d]);
							sum_lambda[d+1] += 0.5*lambda[d+1];
						}
					}
				//}//if( featVal[paramDim].equals("1") )
			}
			
			//System.out.printf("(%.2f%%) ",numError*100.0/numPosSamp);
			
			if( numError == 0 )
				break;
		}
		
		System.out.println();
		
		for( int i=1; i<=paramDim; i++ )
				sum_lambda[i] /= maxIter;
		
		return sum_lambda;
	}

	private double[] run_maxent( Vector<String> samples )
	{
		/*try 
		{
	        Runtime rt = Runtime.getRuntime();
	        String cmd = "/home/yuan/Desktop/stanford-classifier-2011-09-16/src/run.sh";
	       	        
	        //RUN 
	        Process p = rt.exec(cmd);

	        StreamGobbler errorGobbler = new StreamGobbler(p.getErrorStream(), 1);
	        StreamGobbler outputGobbler = new StreamGobbler(p.getInputStream(), 1);

	        errorGobbler.start();
	        outputGobbler.start();

	        int decStatus = p.waitFor();
	        if (decStatus != 0) 
	        {
	          System.out.println("Call to decoder returned " + decStatus
	                + "; was expecting " + 0 + ".");
	          System.exit(30);
	        }
	    } 
		catch (IOException e) 
		{
	        System.err.println("IOException in MertCore.run_decoder(int): " + e.getMessage());
	        System.exit(99902);
	    } 
		catch (InterruptedException e) 
		{
	        System.err.println("InterruptedException in MertCore.run_decoder(int): " + e.getMessage());
	        System.exit(99903);
	    }*/
		System.out.println("--------- Start MaxEnt training ----------");
		System.out.println("Total training samples(positive+negative): "+samples.size());
		
		double[] lambda = new double[paramDim+1];
		String[] args = new String[6];
		args[0] = "-prop";
		//args[1] = "/home/yuan/Desktop/stanford-classifier-2011-09-16/src/examples/iris2007.prop";
		args[1] = "/home/ycao/WS11/nist_zh_en_percep/pro_forward/pro_maxent/prop_file";
		args[2] = "-trainFile";
		args[3] = "null";
		args[4] = "-testFile"; 
		args[5] = "null";
				
		try
		{
			lambda = ColumnDataClassifier.run(args, samples);
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
			
	private double[] run_MegaM( Vector<String> samples )
	{
		double[] lambda = new double[paramDim+1];
		//String root_dir = "/media/Data/JHU/Research/MT discriminative LM training/joshua_expbleu/PRO_test/";
		String root_dir = "/home/ycao/WS11/nist_zh_en_percep/pro_forward/pro_megam/";
		
		System.out.println("------------ Start MegaM training --------------");
		System.out.println("Total training samples(positive+negative): "+samples.size());
		
		try
		{
			//prepare training file for MegaM
			PrintWriter prt = new PrintWriter(new FileOutputStream(root_dir+"megam_train.data"));
			
			for(String line : samples)
			{
				String[] feat = line.split("\\s+");
				
				if( feat[feat.length-1].equals("1") )
					prt.print("1 ");
				else
					prt.print("0 ");
				
				for(int i=0; i<feat.length-1; i++)
					prt.print( (i+1) + " " + feat[i]+" ");  //feat id starts from 1!
				
				prt.println();
			}
			prt.close();
			
			//start running MegaM
			Runtime rt = Runtime.getRuntime();
	        //String cmd = "/home/yuan/tmp_megam_command";
			String cmd = "/home/ycao/WS11/nist_zh_en_percep/pro_forward/pro_megam/megam_command";
	     
	        Process p = rt.exec(cmd);

	        StreamGobbler errorGobbler = new StreamGobbler(p.getErrorStream(), 1);
	        StreamGobbler outputGobbler = new StreamGobbler(p.getInputStream(), 1);

	        errorGobbler.start();
	        outputGobbler.start();

	        int decStatus = p.waitFor();
	        if (decStatus != 0) 
	        {
	        	System.out.println("Call to decoder returned " + decStatus
	                + "; was expecting " + 0 + ".");
	        	System.exit(30);
	        }
	        
	        //read the weights
	        BufferedReader wt = new BufferedReader(new FileReader(root_dir+"megam_weights"));
			String line;
			while( (line = wt.readLine()) != null )
			{
				String val[] = line.split("\\s+");
				lambda[Integer.parseInt(val[0])] = Double.parseDouble(val[1]);
			}
	        
	        File file = new File(root_dir+"megam_train.data");
	        file.delete();
	        file = new File(root_dir+"megam_weights");
	        file.delete();
		}
		catch(IOException exception)
		{
			exception.getStackTrace();
		} 
		catch (InterruptedException e) 
		{
			System.err.println("InterruptedException in MertCore.run_decoder(int): " + e.getMessage());
	        System.exit(99903);;
		}
		
		System.out.println("------------ End MegaM training --------------");
		
		/*
		try 
		{
			Thread.sleep(20000);
		} 
		catch(InterruptedException e) 
		{
		}*/
		
		return lambda;
	}
	
	private double[] run_svm( Vector<String> samples )
	{
		double[] lambda = new double[paramDim+1];
		for( int i=1; i<=paramDim; i++ )
			lambda[i] = 0;
		
		//String root_dir = "/media/Data/JHU/Research/MT discriminative LM training/joshua_expbleu/PRO_test/";
		String root_dir = "/home/ycao/WS11/nist_zh_en_percep/pro_forward/pro_libsvm/";
		
		System.out.println("------------ Start SVM training! --------------");
		System.out.println("Total training samples(positive+negative): "+samples.size());
		
		try
		{
			//prepare training file for MegaM
			PrintWriter prt = new PrintWriter(new FileOutputStream(root_dir+"libsvm_train.data"));
			
			for(String line : samples)
			{
				String[] feat = line.split("\\s+");
				
				if( feat[feat.length-1].equals("1") )
					prt.print("+1 ");
				else
					prt.print("-1 ");
				
				for(int i=0; i<feat.length-1; i++)
					prt.print( (i+1) + ":" + feat[i]+" ");  //feat id starts from 1!
				
				prt.println();
			}
			prt.close();
			
			//start running SVM
			Runtime rt = Runtime.getRuntime();
	        //String cmd = "/home/yuan/tmp_libsvm_command";
			String cmd = "/home/ycao/WS11/nist_zh_en_percep/pro_forward/pro_libsvm/libsvm_command";
	     
	        Process p = rt.exec(cmd);

	        StreamGobbler errorGobbler = new StreamGobbler(p.getErrorStream(), 1);
	        StreamGobbler outputGobbler = new StreamGobbler(p.getInputStream(), 1);

	        errorGobbler.start();
	        outputGobbler.start();

	        int decStatus = p.waitFor();
	        if (decStatus != 0) 
	        {
	        	System.out.println("Call to decoder returned " + decStatus
	                + "; was expecting " + 0 + ".");
	        	System.exit(30);
	        }
	        
	        //read the model file
	        BufferedReader wt = new BufferedReader(new FileReader(root_dir+"libsvm_train.data.model"));
			String line;
			boolean sv_start = false;
			double coef;
			
			while( (line = wt.readLine()) != null )
			{
				if( sv_start )
				{
					String[] val = line.split("\\s+");
					coef = Double.parseDouble(val[0]);
					
					//System.out.print(coef+" ");
					
					for( int i=1; i<val.length; i++ )
					{
						String[] sv = val[i].split(":");
						lambda[ Integer.parseInt(sv[0]) ] += coef * Double.parseDouble(sv[1]);  //index starts from 1
						//System.out.print(Integer.parseInt(sv[0])+" "+Double.parseDouble(sv[1])+" ");
					}
					
					System.out.println();
				}
				
				if( line.equals("SV") )
					sv_start = true;
			}
	        
	        File file = new File(root_dir+"libsvm_train.data");
	        //file.delete();
	        file = new File(root_dir+"libsvm_train.data.model");
	        //file.delete();
		}
		catch(IOException exception)
		{
			exception.getStackTrace();
		} 
		catch (InterruptedException e) 
		{
			System.err.println("InterruptedException in MertCore.run_decoder(int): " + e.getMessage());
	        System.exit(99903);;
		}
		
		System.out.println("------------ End SVM training --------------");
		
		/*
		try 
		{
			Thread.sleep(20000);
		} 
		catch(InterruptedException e) 
		{
		}*/
		
		return lambda;
	}
	
	//from ZMERT
	private void normalizeLambda(double[] origLambda)
	{
		  // private String[] normalizationOptions;
	      // How should a lambda[] vector be normalized (before decoding)?
	      //   nO[0] = 0: no normalization
	      //   nO[0] = 1: scale so that parameter nO[2] has absolute value nO[1]
	      //   nO[0] = 2: scale so that the maximum absolute value is nO[1]
	      //   nO[0] = 3: scale so that the minimum absolute value is nO[1]
	      //   nO[0] = 4: scale so that the L-nO[1] norm equals nO[2]

		int normalizationMethod = (int)normalizationOptions[0];
	    double scalingFactor = 1.0;
	    if (normalizationMethod == 0) 
	    {
	    	scalingFactor = 1.0;
	    } 
	    else if (normalizationMethod == 1) 
	    {
	    	int c = (int)normalizationOptions[2];
	    	scalingFactor = normalizationOptions[1]/Math.abs(origLambda[c]);
	    } 
	    else if (normalizationMethod == 2) 
	    {
	    	double maxAbsVal = -1;
	    	int maxAbsVal_c = 0;
	    	for (int c = 1; c <= paramDim; ++c) 
	    	{
	    		if (Math.abs(origLambda[c]) > maxAbsVal) 
	    		{
	    			maxAbsVal = Math.abs(origLambda[c]);
	    			maxAbsVal_c = c;
	    		}
	    	}
	    	scalingFactor = normalizationOptions[1]/Math.abs(origLambda[maxAbsVal_c]);

	    } 
	    else if (normalizationMethod == 3) 
	    {
	    	double minAbsVal = PosInf;
	    	int minAbsVal_c = 0;
	    	
	    	for (int c = 1; c <= paramDim; ++c) 
	    	{
	    		if (Math.abs(origLambda[c]) < minAbsVal) 
	    		{
	    			minAbsVal = Math.abs(origLambda[c]);
	    			minAbsVal_c = c;
	    		}
	    	}
	    	scalingFactor = normalizationOptions[1]/Math.abs(origLambda[minAbsVal_c]);

	    } 
	    else if (normalizationMethod == 4) 
	    {
	    	double pow = normalizationOptions[1];
	    	double norm = L_norm(origLambda,pow);
	    	scalingFactor = normalizationOptions[2]/norm;
	    }

	    for (int c = 1; c <= paramDim; ++c) 
	    {
	    	origLambda[c] *= scalingFactor;
	    }
	}
	
	//from ZMERT
	private double L_norm(double[] A, double pow)
	{
	    // calculates the L-pow norm of A[]
	    // NOTE: this calculation ignores A[0]
		double sum = 0.0;
	    for (int i = 1; i < A.length; ++i) 
	      sum += Math.pow(Math.abs(A[i]),pow);
	    	
	    return Math.pow(sum,1/pow);
	}
	
	private EvaluationMetric evalMetric;
	private Vector<String> output;
	private double[] initialLambda;
	private double[] finalLambda;
	private double finalScore;
	private double[] normalizationOptions;
	private HashMap<String, String>[] feat_hash;
	private HashMap<String, String>[] stats_hash;
	private Random randgen;
	private int j;
	private int paramDim;
	private int sentNum;
	private int Tau; //size of sampled candidate set(say 5000)
	private int Xi; //choose top Xi candidates from sampled set(say 50)
	private double metricDiff; //metric difference threshold(to select the qualified candidates)
	private String optAlgorithm; //optimization algorithm
	
	private final static double NegInf = (-1.0 / 0.0);
	private final static double PosInf = (+1.0 / 0.0);
}

class ValueComparator implements Comparator
{
	Map base;
	public ValueComparator(Map base) 
	{
		this.base = base;
	}

	public int compare(Object a, Object b) 
	{
		if((Double)base.get(a) <= (Double)base.get(b))
			return 1;
		else if((Double)base.get(a) == (Double)base.get(b))
			return 0;
		else 
			return -1;
  }	
}
