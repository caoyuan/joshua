package joshua.discriminative.training.risk_annealer.hypergraph.kbest_hg;


import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.logging.Level;
import java.util.logging.Logger;

import joshua.corpus.vocab.SymbolTable;
import joshua.decoder.BLEU;
import joshua.decoder.NbestMinRiskReranker;
import joshua.decoder.hypergraph.HGNode;
import joshua.decoder.hypergraph.HyperEdge;
import joshua.decoder.hypergraph.KBestExtractor;
import joshua.discriminative.training.risk_annealer.hypergraph.FeatureForest;

/** Given a hg
* @author Zhifei Li, <zhifei.work@gmail.com>
* @version $LastChangedDate: 2008-10-20 00:12:30 -0400 $
*/

public class KbestHGGradientComputer {
	// Congiruation parameters.
	private double temperature; 
	private KBestExtractor kbestExtrator;
	private double gainFactor;
	private double scale;
	private double[] weights;
	//## for BLEU
	private static int bleuOrder = 4;
	private static boolean doNgramClip = true;
	private boolean useShortestRefLen = false;
	private boolean useLogBleu = false;
	//============== google linear corpus gain
	private boolean useGoogleLinearCorpusGain = false;
	private double[] linearCorpusGainThetas; //weights in the Goolge linear corpus gain function
	
	private FeatureForest hg;
	private String[] referenceSentences;
	
	//latest value
	private double entropy;
	private double risk;
	private double functionValue;

	private static final Logger logger = Logger.getLogger(KbestHGGradientComputer.class.getName());
	
	public KbestHGGradientComputer(SymbolTable symbolTbl, double temperature,
			double gainFactor, double scale, double[] weights, boolean useGoogleLinearCorpusGain,
			double[] linearCorpusGainThetas) {
		kbestExtrator = new KBestExtractor(symbolTbl, false, false, false, false, false, false);
		this.temperature = temperature;
		this.gainFactor = gainFactor;
		this.scale = scale;
		this.weights = weights;
		this.useGoogleLinearCorpusGain = useGoogleLinearCorpusGain;
		this.linearCorpusGainThetas = linearCorpusGainThetas;
		
		//NEWLY ADDED
		logger.setLevel(Level.OFF);
	}
	
	public final void setReferenceSentences(String[] referenceSentences){
		this.referenceSentences = referenceSentences;
	}
		
	public final void setTemperature(double temperature){
		this.temperature = temperature;
	}
	
	protected final double getTemperature(){
		return this.temperature;
	}
	 
	public double getEntropy(){
		return this.entropy;
	}
	 
	public double getRisk(){
		return this.risk;	
	}
	
	public double getFuncVal(){
		return functionValue;//this.risk - this.temperature*this.entropy;		
	}
	
	public void setHyperGraph(FeatureForest hg){
		this.hg = hg;
	}
	 
	public HashMap<Integer, Double> computeGradientForTheta(){
		// Do inference.
		List<Map<Integer, Double>> featureTbls = new ArrayList<Map<Integer, Double>>();
		List<Double> nbestProbs = new ArrayList<Double>();		
		List<Double> gainWithRespectToRef = new ArrayList<Double>(); 
		Inference(hg, referenceSentences, featureTbls, nbestProbs, gainWithRespectToRef); 
		
		//normalize the probability distribution
		NbestMinRiskReranker.computeNormalizedProbs(nbestProbs, scale);
		
		this.entropy = computeEntropy(nbestProbs);
		this.risk = - computeExpectedGain(gainWithRespectToRef, nbestProbs);
		this.functionValue = this.risk - this.temperature*this.entropy;		
		//System.out.println("Entropy=" + entropy + "; risk=" + risk + "; function=" + functionValue);
		
		// Compute feature expectation.
		Map<Integer, Double> expectedFeatureValues = new HashMap<Integer, Double> ();
		for(int i=0; i < nbestProbs.size(); i++){
			double prob = nbestProbs.get(i);
			for(Map.Entry<Integer, Double> feature : featureTbls.get(i).entrySet()){
				Double oldVal = expectedFeatureValues.get(feature.getKey());
				if (oldVal == null)
					oldVal = 0.0;				
				expectedFeatureValues.put(feature.getKey(), oldVal + prob * feature.getValue());
			}
		}
		
		// Compute gadient.
		HashMap<Integer, Double> gradient = new HashMap<Integer, Double>();
		for (int i = 0; i < nbestProbs.size(); ++i) {	
			double prob = nbestProbs.get(i);			
			double gain = gainWithRespectToRef.get(i)*gainFactor;
			double entropyFactor;
			if(prob == 0)
				entropyFactor = -temperature*(0 + 1);//+TH(P); log(0)=0 as otherwise not well-defined 
			else
				entropyFactor = -temperature*(Math.log(prob) + 1);//+TH(P)
			// It is critical to iterate over expectedFeatureValues, not featureTbls.get(i), because
			// the feature (that fires in expectedFeatureValues but not on a given tree) still needs 
			// to be considered for the given tree. On the other hand, if a feature does not even
			// fire in expectedFeatureValues, the gradient is zero. That's why we do not need to iterate
			// over the global feature set.
			Map<Integer, Double> feat_tbl = featureTbls.get(i);
			for(Map.Entry<Integer, Double> featureInExpTbl : expectedFeatureValues.entrySet()){		
				Double featVal = feat_tbl.get(featureInExpTbl.getKey());
				if (featVal == null)
					featVal = 0.0;
				double common = scale*prob*(featVal - featureInExpTbl.getValue());
				double sentGradient = - common * (gain + entropyFactor); // risk - T * Entropy				
				Double oldVal = gradient.get(featureInExpTbl.getKey());
				if (oldVal == null)
					oldVal = 0.0;				
				gradient.put(featureInExpTbl.getKey(), oldVal + sentGradient);
			}
		}
		if (gradient.size() != expectedFeatureValues.size()) {
			logger.severe("gradient.size() != expectedFeatureValues.size()");
			System.exit(1);	
		}
		return gradient;
	}
	
	private void Inference(FeatureForest ff,
			String[] referenceSentences,
			List<Map<Integer, Double>> featureTbls,
			List<Double> nbestLogProbs,
			List<Double> gainWithRespectToRef) {
		featureTbls.clear();
		nbestLogProbs.clear();
		gainWithRespectToRef.clear();
		logger.info("number of trees is " + ff.goalNode.hyperedges.size());
		kbestExtrator.resetState();
		for (int i = 0; i < ff.goalNode.hyperedges.size(); ++i) {
			HyperEdge edge = ff.goalNode.hyperedges.get(i);
			if (edge.getAntNodes().size() != 1) {
				logger.severe("number of ant node is not one");
				System.exit(1);
			}
			//============== Collect features in tree.
			Map<Integer, Double> treeFeatures = new HashMap<Integer, Double>();
			featureTbls.add(treeFeatures);
			collectFeaturesInTree(ff, edge, treeFeatures);
			double logP = 0;
			for(Map.Entry<Integer, Double> feature : treeFeatures.entrySet()){
				logP += weights[feature.getKey()] * feature.getValue();
			}		
			nbestLogProbs.add(logP); // note that scale is added in computeNormalizedProbs.
			
			//============== Collect loss/gain.
			String translation = kbestExtrator.getKthHyp(edge.getAntNodes().get(0), 1, -1, null, null);
			kbestExtrator.resetState();
			double gain=0;
			if(useGoogleLinearCorpusGain){
				int hypLength = translation.split("\\s+").length;
				HashMap<String, Integer> refereceNgramTable = 
					BLEU.constructMaxRefCountTable(referenceSentences, bleuOrder);
				HashMap<String, Integer> hypNgramTable = BLEU.constructNgramTable(translation, bleuOrder);
				gain = BLEU.computeLinearCorpusGain(linearCorpusGainThetas, 
						hypLength, hypNgramTable,  refereceNgramTable); 
			} else{
				gain = BLEU.computeSentenceBleu(referenceSentences, translation, doNgramClip,
						bleuOrder, useShortestRefLen);
			}
			
			if(useLogBleu){
				if(gain == 0)
					gainWithRespectToRef.add(0.0);//log0=0
				else
					gainWithRespectToRef.add(Math.log(gain));
			} else
				gainWithRespectToRef.add(gain);
		}
	}
	
	private void collectFeaturesInTree(FeatureForest ff, HyperEdge rootEdge, Map<Integer, Double> treeFeatures) {
		// Accumulate features at each edge
		Map<Integer, Double> edgeFeatures = ff.featureExtraction(rootEdge, null);
		for (Map.Entry<Integer, Double> feature : edgeFeatures.entrySet()) {
			Double oldVal = treeFeatures.get(feature.getKey());
			if (oldVal == null)
				oldVal = 0.0;
			treeFeatures.put(feature.getKey(), oldVal + feature.getValue());
		}
		
		// Recursive call.
		if (null != rootEdge.getAntNodes()) {
			for (HGNode node : rootEdge.getAntNodes()) {
				if (node.hyperedges.size() != 1) {
					logger.severe("The node has more than one edge, this cannot be true for a kbest hg");
					System.exit(1);
				}
				collectFeaturesInTree(ff, node.hyperedges.get(0), treeFeatures);
			}
		}
	}
	
	private double computeEntropy(List<Double> nbestProbs) {
		double entropy =0;
		double tSum=0;
		for(double prob : nbestProbs){
			if(prob!=0)//log0 is not well defined
				entropy -= prob*Math.log(prob);//natural base
			//if(Double.isNaN(entropy)){System.out.println("entropy becomes NaN, must be wrong; prob is " + prob ); System.exit(1);}
			tSum+=prob;
		}
//		sanity check
		if(Math.abs(tSum-1.0)>1e-4){
			System.out.println("probabilities not sum to one, must be wrong");
			System.exit(1);}
		if(Double.isNaN(entropy)){
			System.out.println("entropy is NaN, must be wrong");
			System.exit(1);}
		if(entropy<0 || entropy > Math.log(nbestProbs.size()+ 1e-2)){
			System.out.println("entropy is negative or above upper bound, must be wrong; " + entropy);
			System.exit(1);}
		//System.out.println("entropy is: " + entropy);
		return entropy;
	}
	
	private double computeExpectedGain(List<Double> nbestGains, List<Double> nbestProbs) {
		double expectedGain = 0;
		for(int i=0; i<nbestGains.size(); i++){
			double gain = nbestGains.get(i);
			double trueProb = nbestProbs.get(i);
			expectedGain += trueProb*gain;
		}
				
		//sanity check
		if(Double.isNaN(expectedGain)){
			System.out.println("expected_gain isNaN, must be wrong");
			System.exit(1);
		}
		if(useGoogleLinearCorpusGain == false){
			if(useLogBleu){
				if(expectedGain>1e-2){
					System.out.println("Warning: expected_gain is not smaller than zero when using logBLEU, must be wrong: " + expectedGain);
					System.exit(1);
				}
			} else {
				if(expectedGain<-(1e-2) || expectedGain > 1+1e-2){
					System.out.println("Warning: expected_gain is not within [0,1], must be wrong: " + expectedGain);
					System.exit(1);
				}
			}
		}
		return expectedGain;
	} 
}
