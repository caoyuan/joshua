/**
 * 
 */
package joshua.discriminative.training.risk_annealer.hypergraph.parallel;

import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.BlockingQueue;

import joshua.corpus.vocab.SymbolTable;
import joshua.discriminative.semiring_parsing.MinRiskDAFuncValSemiringParser;
import joshua.discriminative.semiring_parsing.MinRiskDAGradientSemiringParser;
import joshua.discriminative.semiring_parsingv2.applications.min_risk_da.MinRiskDADenseFeaturesSemiringParser;
import joshua.discriminative.training.parallel.Consumer;
import joshua.discriminative.training.risk_annealer.hypergraph.FeatureForest;
import joshua.discriminative.training.risk_annealer.hypergraph.HGAndReferences;
import joshua.discriminative.training.risk_annealer.hypergraph.HGRiskGradientComputer;
import joshua.discriminative.training.risk_annealer.hypergraph.MRConfig;
import joshua.discriminative.training.risk_annealer.hypergraph.RiskAndFeatureAnnotationOnLMHG;
import joshua.discriminative.training.risk_annealer.hypergraph.kbest_hg.KbestHGGradientComputer;

public class GradientConsumer extends Consumer<HGAndReferences> {	
	private final HGRiskGradientComputer gradientComputer;
	// Different values of gradient computer.
	MinRiskDADenseFeaturesSemiringParser gradientSemiringParserV2;
	KbestHGGradientComputer kbestHGGradientComputer;
	
	MinRiskDAGradientSemiringParser gradientSemiringParserV1;
	MinRiskDAFuncValSemiringParser funcValSemiringParserV1; 
	
	RiskAndFeatureAnnotationOnLMHG  riskAnnotator;
	double[] weightsForTheta;
	
	double scalingFactor;
	boolean shouldComputeGradientForScalingFactor;
	double temperature;
	
	boolean useSemiringV2=true;
	
	SymbolTable symbol;
	
	//static private Logger logger = Logger.getLogger(GradientConsumer.class.getSimpleName()); 
	
	public GradientConsumer(boolean useSemiringV2, HGRiskGradientComputer gradientComputer, BlockingQueue<HGAndReferences> q, double[] weightsForTheta,
			RiskAndFeatureAnnotationOnLMHG riskAnnotator, double temperature,
			double scalingFactor, boolean shouldComputeGradientForScalingFactor,
			SymbolTable symbol) {		
		super(q);
		this.useSemiringV2 = useSemiringV2;
		this.gradientComputer = gradientComputer;
		 
		this.weightsForTheta = weightsForTheta;
		
		this.riskAnnotator = riskAnnotator;
	
		this.temperature = temperature;
		this.scalingFactor = scalingFactor;
		this.shouldComputeGradientForScalingFactor = shouldComputeGradientForScalingFactor;
		
		if (MRConfig.use_kbest_hg && MRConfig.brute_force_for_kbest_hg) {
			//System.out.println("linearCorpusGainThetas=" + riskAnnotator.linearCorpusGainThetas());
			this.kbestHGGradientComputer = new KbestHGGradientComputer(symbol, temperature, 
					MRConfig.gainFactor, scalingFactor, weightsForTheta,
					MRConfig.useGoogleLinearCorpusGain, riskAnnotator.linearCorpusGainThetas());
		} else {
			if(useSemiringV2){
				//System.out.println("----------------useSemiringV2");
				this.gradientSemiringParserV2 =  new MinRiskDADenseFeaturesSemiringParser(this.temperature);			
			} else {
				//System.out.println("----------------useSemiringV1");
				this.gradientSemiringParserV1 = new MinRiskDAGradientSemiringParser(1, 0, scalingFactor, temperature);			
		    	this.funcValSemiringParserV1 = new MinRiskDAFuncValSemiringParser(1, 0, scalingFactor, temperature);			
			}
		}
	}

	@Override
	public void consume(HGAndReferences hgAndRefs) {		
		FeatureForest fForest = riskAnnotator.riskAnnotationOnHG(hgAndRefs.hg, hgAndRefs.referenceSentences);
		fForest.setFeatureWeights(weightsForTheta);
		fForest.setScale(scalingFactor);
		/** Based on a model and a test hypergraph 
		 * (which provides the topology and feature/risk annotation), 
		 *  compute the gradient and function value.
		 **/
		if (MRConfig.use_kbest_hg && MRConfig.brute_force_for_kbest_hg) {
			consumeHelperForKbestHG(fForest, hgAndRefs.referenceSentences);
		} else {
			if(this.useSemiringV2){
				consumeHelperV2(fForest);
			} else{
				consumeHelperV1(fForest);
			}
		}
	}	
	
	private void consumeHelperV1(FeatureForest fForest){
		gradientSemiringParserV1.setHyperGraph(fForest);
		HashMap<Integer, Double> gradients = gradientSemiringParserV1.computeGradientForTheta();
		
		double gradientForScalingFactor = 0;
		if(shouldComputeGradientForScalingFactor)
			gradientForScalingFactor -= computeGradientForScalingFactor(gradients, weightsForTheta, scalingFactor);//we are maximizing, instead of minizing
		
		//== compute function value	
		funcValSemiringParserV1.setHyperGraph(fForest);
		double funcVal = funcValSemiringParserV1.computeFunctionVal();//risk-T*entroy
		double risk = funcValSemiringParserV1.getRisk();
		double entropy = funcValSemiringParserV1.getEntropy();
		
		//== accumulate gradient and function value //risk-T*entroy
		this.gradientComputer.accumulateGradient(gradients, gradientForScalingFactor, funcVal, risk, entropy);
		//logger.info("=====consumed one sentence ");
	}
	
	private void consumeHelperV2(FeatureForest fForest){
//		@todo: we should check if hg_test is a feature forest or not
		gradientSemiringParserV2.setHyperGraph(fForest);

		//== compute gradient and function value
		HashMap<Integer, Double> gradients = gradientSemiringParserV2.computeGradientForTheta();
		
		double gradientForScalingFactor = 0;
		if(this.shouldComputeGradientForScalingFactor)
			gradientForScalingFactor = computeGradientForScalingFactor(gradients, weightsForTheta, scalingFactor);
		
		double funcVal = gradientSemiringParserV2.getFuncVal();//risk-T*entroy
		double risk = gradientSemiringParserV2.getRisk();
		double entropy = gradientSemiringParserV2.getEntropy();
		
		//== accumulate gradient and function value: //risk-T*entroy
		this.gradientComputer.accumulateGradient(gradients, gradientForScalingFactor, funcVal, risk, entropy);
		//logger.info("=====consumed one sentence ");
	}
	
	// This computes gradients on the kbest hypergraph, which does not requires that the loss function
	// to be edge decomposable, nor does it require to use local features only (though todo below).
	//
	// TODO(zfli): we still use only local features, since we directly use the features from FeatureForest.
	// But, we will ignores the edge loss in FeatureForest, and instead compute the sentence-wide loss.
	private void consumeHelperForKbestHG(FeatureForest fForest, String[] referenceSentences){
//		@todo: we should check if hg_test is a feature forest or not
		kbestHGGradientComputer.setHyperGraph(fForest);
		kbestHGGradientComputer.setReferenceSentences(referenceSentences);

		//== compute gradient and function value
		HashMap<Integer, Double> gradients = kbestHGGradientComputer.computeGradientForTheta();
		
		double gradientForScalingFactor = 0;
		if(this.shouldComputeGradientForScalingFactor)
			gradientForScalingFactor = computeGradientForScalingFactor(gradients, weightsForTheta, scalingFactor);
		
		double funcVal = kbestHGGradientComputer.getFuncVal();//risk-T*entroy
		double risk = kbestHGGradientComputer.getRisk();
		double entropy = kbestHGGradientComputer.getEntropy();
		
		//== accumulate gradient and function value: //risk-T*entroy
		gradientComputer.accumulateGradient(gradients, gradientForScalingFactor, funcVal, risk, entropy);
		
		//logger.info("=====consumed one sentence ");
	}
	
	@Override
	public boolean isPoisonObject(HGAndReferences x) {
		return (x.hg==null);
	}

	private double computeGradientForScalingFactor(HashMap<Integer, Double>  gradientForTheta, 
			double[] weightsForTheta, double scale){    	
		double gradientForScale = 0;
		for(Map.Entry<Integer, Double> feature : gradientForTheta.entrySet()){		 
			gradientForScale +=   weightsForTheta[feature.getKey()] *  feature.getValue();
		}
		gradientForScale /= scale;
		//System.out.println("****gradientForScale" + gradientForScale + "; scale: " + scale );		
		
		if(Double.isNaN(gradientForScale)){
			System.out.println("gradient value for scaling is NaN"); 
			System.exit(1);
		} 
		//System.out.println("Gradient for scale is : " + gradientForScale);
		return gradientForScale;
	}
}