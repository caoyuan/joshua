package joshua.discriminative.training.expbleu_yuan;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.BlockingQueue;
import java.util.logging.Logger;

import joshua.corpus.vocab.SymbolTable;
import joshua.discriminative.feature_related.feature_template.FeatureTemplate;
import joshua.discriminative.training.expbleu_yuan.parallel.GradientConsumer;
import joshua.discriminative.training.parallel.ProducerConsumerModel;
import joshua.discriminative.training.risk_annealer.GradientComputer;
import joshua.discriminative.training.risk_annealer.hypergraph.HGAndReferences;
import joshua.discriminative.training.risk_annealer.hypergraph.HyperGraphFactory;
import joshua.discriminative.training.risk_annealer.hypergraph.MRConfig;
import joshua.discriminative.training.risk_annealer.hypergraph.parallel.HGProducer;
import joshua.util.Regex;

public class ExpbleuGradientComputer extends GradientComputer {
	/*
	 * In this class, we need to compute the gradient for theta and the function value given 
	 * current theta. We use temperature 0 and no scaling.(maybe added later)
	 * Store function value in functionValue
	 * Store gradient values in gradientsForTheta
	 */
	private int numSentence;     
	private boolean fixFirstFeature = false;

	HyperGraphFactory hgFactory;  

	private  double sumGain = 0; //negative risk

	int numCalls = 0;	    
	int maxNumHGInQueue = 100;
	int numThreads = 5;
	
	//constant 
    private double N = 100;

	boolean useSemiringV2 = true;

	// feature related
	SymbolTable symbolTbl;
	HashMap<String, Integer> featureStringToIntegerMap;
	List<FeatureTemplate> featTemplates;	  

	boolean haveRefereces = true;	    
	double minFactor = 1.0; //minimize conditional entropy: 1; minimum risk: -1
	int numFeats;
	
	// Ngram Matches Stats and gradients
	//CORPUS LEVEL!
	double [] ngramMatches = new double[5];  
	private double minlen = 0; 
	ArrayList<ArrayList<Double>> ngramMatchesGradients = new ArrayList<ArrayList<Double>>();  //THE GRADIENT OF NGRAMMATCH WRT EACH FEATURE WEIGHT
	
	private int consumed = 0;
	/** Logger for this class. */
	static final private Logger logger = 
		Logger.getLogger(ExpbleuGradientComputer.class.getSimpleName());
	
	public ExpbleuGradientComputer(int numFeatures, double gainFactor,
			double scalingFactor, double temperature,
			boolean shouldComputeGradientForScalingFactor, boolean useSemiringV2, int numSentence, HyperGraphFactory hgFactory,  SymbolTable symbolTbl, HashMap<String, Integer> featureStringToIntegerMap, List<FeatureTemplate> featTemplates, boolean haveRefereces, int maxNumHGInQueue, int numThreads) {
		super(numFeatures, gainFactor, scalingFactor, temperature,
				shouldComputeGradientForScalingFactor);
		this.useSemiringV2 = useSemiringV2;
		this.numSentence = numSentence;
		
//		System.out.println("Pause . . . . . . . . . .");
//		try {
//			Thread.sleep(2000);
//		} 
//		catch(InterruptedException e) {
//		}
		
		this.hgFactory = hgFactory;
		this.maxNumHGInQueue = maxNumHGInQueue;  //PRUNING THRESHOLD
		this.numThreads = numThreads;
		// System.out.println("use HGRiskGradientComputer====");

		this.symbolTbl = symbolTbl;               

		//PROCESS FEATURE RELATED DATA STRUCTURE
		this.featureStringToIntegerMap = featureStringToIntegerMap;
		this.featTemplates = featTemplates;
		this.haveRefereces = haveRefereces;
		this.numFeats = featureStringToIntegerMap.size();
		
		for(int i = 0; i < 5; ++i){
			this.ngramMatches[i] = 0;
			
			//INITIALIZE THETA FOR EACH FEATURE
			ArrayList<Double> row = new ArrayList<Double>(10);
			for(int j = 0; j < this.numFeats ; ++j){
				row.add(Double.valueOf(0));
			}
			
			//THE GRADIENT W.R.T EATH THETA 
			this.ngramMatchesGradients.add(row);
		}

	}

	//@Override
	public void printLastestStatistics() {
		// TODO Auto-generated method stub

	}

	//PARELLEL VERSION - CALLING NON-PARELLEL VERSION
	@Override
	public void reComputeFunctionValueAndGradient(double[] theta) {
		// initialize all counts to 0
		for(int i = 0; i < 5; ++i){
			this.ngramMatches[i] = 0;
			
			//THE THETA FOR EACH FEATURE
			for(int j = 0; j < this.numFeats; ++j ){
				this.ngramMatchesGradients.get(i).set(j, 0.0);  //SET ALL NGRAM GRADIENTS TO 0
			}
		}
		
		this.minlen = 0; 
		this.functionValue = 0; 
		for(int i = 0; i < this.numFeats; ++i){
			this.gradientsForTheta[i] = 0;  //GRADIENT OF THE ENTIRE FUNCTION (EXP_BLEU)  
		}
				
		if(this.numThreads == 1){
			reComputeFunctionValueAndGradientNonparellel(theta);  //NON-PARELLEL VERSION
		} else{
			BlockingQueue<HGAndReferences> queue = new ArrayBlockingQueue<HGAndReferences>(maxNumHGInQueue);
			this.hgFactory.startLoop();
			System.out.println("Compute function value and gradients for expbleu");
			System.out.print("[");
			
			//SETTINGS FOR EACH THREAD
			HGProducer producer = new HGProducer(hgFactory, queue, numThreads, numSentence);
			List<GradientConsumer> consumers = new ArrayList<GradientConsumer>();
			for(int i = 0; i < this.numThreads; ++i)
			{
				GradientConsumer consumer = new GradientConsumer( queue, this.featTemplates, this.featureStringToIntegerMap, theta, this.symbolTbl,this);
				consumers.add(consumer);
			}
			//== create model, and start parallel computing
			ProducerConsumerModel<HGAndReferences, HGProducer, GradientConsumer> 
			model =	new ProducerConsumerModel<HGAndReferences, HGProducer, GradientConsumer>(queue, producer, consumers);

			//RUN IN PARALLEL
			model.runParallel();
			this.consumed = 0;
			System.out.print("]\n");
			this.hgFactory.endLoop();
		}

		finalizeFunAndGradients();
	}

	//COMPUTE THE FINAL FUNCTION VALUE AND DERIVATIVES FOR EACH PARAMETER
	private void finalizeFunAndGradients() {
		//COMPUTE THE FUNCTION VALUE
		
		//M_i
		for(int i = 0; i < 4; ++i){
			this.functionValue += 1.0/4.0 * Math.log(ngramMatches[i]);
		}
		
		//K_i
		for(int i = 0; i < 4; ++i){
			this.functionValue -= 1.0/4.0 * Math.log(ngramMatches[4] - i * this.numSentence );
		}
		
		double ATAN_SCALE = 1.8;  //EMPIRICAL OPTIMAL VALUE
		double ATAN_CONST = 24/Math.PI;
		double ratio = this.ngramMatches[4]/this.minlen;  // H/R NOT R/H HERE! (to make derivative easier)
		double arctan = -1 * Math.atan(ATAN_SCALE*ratio);  //PRE_COMPUTE FOR SPEEDING UP
		
		double t = Math.atan(ATAN_SCALE*ratio)*ATAN_CONST - 6;
		this.functionValue += -1 * Math.log(1 + Math.exp(-1*t));

		System.out.println("Using sigmoid function as approximation to brevity ...");
		
		//START TO COMPUTE THE DERIVATIVE
		for(int i = 0; i < this.numFeatures ; ++i)
		{
			for(int j = 0; j < 4; ++j)
			{
				
				this.gradientsForTheta[i] += 1.0/4.0/ngramMatches[j]*ngramMatchesGradients.get(j).get(i);
			}
			for(int j = 0; j < 4; ++j)
			{
				this.gradientsForTheta[i] -= 1.0/4.0/(ngramMatches[4] - j*this.numSentence)*ngramMatchesGradients.get(4).get(i);
			}
			
			double d_ratio = this.ngramMatchesGradients.get(4).get(i) / this.minlen;  //d(H/R) 
			//System.out.println(d_ratio);
			
			double d_brev = ( Math.exp(arctan*ATAN_CONST+6) * 
							( -1*ATAN_CONST*ATAN_SCALE*d_ratio / ( 1+(ATAN_SCALE*ATAN_SCALE*ratio*ratio))) )/
							( 1+Math.exp(arctan*ATAN_CONST+6) ) * -1;
			
			this.gradientsForTheta[i] += d_brev;
		}
			
// Ziyuan's version
//		double x = 1 - this.minlen/this.ngramMatches[4];  //1-R/H
//		this.functionValue += 1/(Math.exp(N*x) + 1) * x;
//		
//		//System.out.println("--------------------- " + 1/(Math.exp(N*x) + 1) * x + "  " + x);
//		
//		//COMPUTE THE DERIVATIVE
//		double y;
//		if(x > 0){
//			y = ((1 - N * x)*myexp(-N*x) + myexp(-2*N*x))/(myexp(-N*x) + 1)/(myexp(-N*x)+1);
//		}else{
//			y = ((1 - N * x)*myexp(N*x) + 1)/(myexp(N*x) + 1)/(myexp(N*x)+1);
//		}
//		for(int i = 0; i < this.numFeatures ; ++i){
//			for(int j = 0; j < 4; ++j){
//				
//				this.gradientsForTheta[i] += 1.0/4.0/ngramMatches[j]*ngramMatchesGradients.get(j).get(i);
//			}
//			for(int j = 0; j < 4; ++j){
//				this.gradientsForTheta[i] -= 1.0/4.0/(ngramMatches[4] - j*this.numSentence)*ngramMatchesGradients.get(4).get(i);
//			}
//			double dx =  this.minlen/this.ngramMatches[4]/this.ngramMatches[4]*this.ngramMatchesGradients.get(4).get(i);
////			System.out.println(dx);
//			this.gradientsForTheta[i] += y*dx;
//		}
		
		this.logger.info("Function Value :" + this.functionValue);
		String diffinfo = "Derivatives :";
		for(int i = 0; i < MRConfig.printFirstN; ++i){
			diffinfo += " ";
			diffinfo += this.gradientsForTheta[i];
		}
		this.logger.info(diffinfo);
	}

	//NON-PARELLEL VERSION
	public void reComputeFunctionValueAndGradientNonparellel(double[] theta){		
		this.hgFactory.startLoop();  //START READING HYPERGRAPH
		System.out.println("Compute function value and gradients for expbleu !!!");
		System.out.print("[");
		
		for(int cursent = 0; cursent < this.numSentence; ++ cursent){  //FOR EACH SENTENCE
			HGAndReferences hgres = this.hgFactory.nextHG();  //GET ITS HYPER GRAPH AND REFERENCES
			
			//GET THE MIN LENGTH OF ALL ITS REFS, FOR LENGTH RATIO COMPUTATION
			int minlenForOne = 10000;
			for(String ref : hgres.referenceSentences){
				String [] words = Regex.spaces.split(ref);
				if(words.length < minlenForOne)
					minlenForOne = words.length;
			}
			this.minlen += 1.0 * minlenForOne;  //ACCUMULATE MIN LENGTH
			
			//START SEMERING PARSING
			ExpbleuSemiringParser parser =  new ExpbleuSemiringParser(
					hgres.referenceSentences,
					this.featTemplates,
					this.featureStringToIntegerMap,
					theta,
					new HashSet<String>(this.featureStringToIntegerMap.keySet()),
					this.symbolTbl);

			parser.setHyperGraph(hgres.hg);  //LOAD THE HYPERGRAPH FOR THIS SENTENCE
			
			//****** COMPUTE THE NGRAM AND ITS GRADIENT ******
			parser.parseOverHG();
			double [] matches = parser.getNgramMatches();  //RETURN (EXPECTED)NGRAM MATCH FOR THIS SENTENCE
			
			for(int i = 0; i < 5; ++i)
			{
				ngramMatches[i] += matches[i];  //ACCUMULATE NGRAM MATCH

				double[] matchGradient = parser.getGradients(i);  //ACCUMULATE THE GRAEIDNT FOR NGRAM
				for(int j = 0; j < this.numFeatures ; ++j)
				{
					ngramMatchesGradients.get(i).set(j, ngramMatchesGradients.get(i).get(j) + matchGradient[j]);
				}
			}
			
//			System.out.println(matches[0]);
			if(cursent % 100 == 0){
				System.out.print(".");
			}
		}
		System.out.print("]\n");
		this.hgFactory.endLoop();
	}
	
	//ACCUMULATING NGRAMS AND GRADIENTS USED BY PARALLEL VERSION
	public synchronized void accumulate(ArrayList<ArrayList<Double>> ngramMatchesGradients, double [] matchs, double minlen)
	{
		for(int i = 0; i < 5; ++i){
			this.ngramMatches[i] += matchs[i];
			for(int j = 0; j < this.numFeats; ++j){
				this.ngramMatchesGradients.get(i).set(j, 
							this.ngramMatchesGradients.get(i).get(j) + ngramMatchesGradients.get(i).get(j));
			}
		}
		this.minlen += minlen;
		consumed ++;
		if(consumed % 100 == 0){
			System.out.print(".");
		}
	}

	//Ziyuan's version
		private double myexp(double x){
			if(Double.isInfinite(Math.exp(x))){
				return 0;
			}
			else{
				return Math.exp(x);
			}
		}
	
}
