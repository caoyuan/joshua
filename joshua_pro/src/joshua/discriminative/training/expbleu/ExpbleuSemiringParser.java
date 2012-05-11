package joshua.discriminative.training.expbleu;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;

import joshua.corpus.vocab.SymbolTable;
import joshua.decoder.BLEU;
import joshua.decoder.ff.lm.NgramExtractor;
import joshua.decoder.hypergraph.HGNode;
import joshua.decoder.hypergraph.HyperEdge;
import joshua.discriminative.feature_related.feature_template.FeatureTemplate;
import joshua.discriminative.semiring_parsingv2.DefaultIOParserWithXLinearCombinator;
import joshua.discriminative.semiring_parsingv2.SignedValue;
import joshua.discriminative.semiring_parsingv2.pmodule.ExpectationSemiringPM;
import joshua.discriminative.semiring_parsingv2.pmodule.ListPM;
import joshua.discriminative.semiring_parsingv2.pmodule.SparseMap;
import joshua.discriminative.semiring_parsingv2.semiring.ExpectationSemiring;
import joshua.discriminative.semiring_parsingv2.semiring.LogSemiring;
import joshua.discriminative.training.risk_annealer.hypergraph.MRConfig;

/*
 * ExpBleu Semiring compute <p_e,p_e*m_e,\Delta P_e, (\Delta P_e) * m_e>, where m_e is the ngram matches function. 
 * Static Methods Used from MRConfig, SignedValue
 */
public class ExpbleuSemiringParser extends DefaultIOParserWithXLinearCombinator<ExpectationSemiring<LogSemiring,NgramMatchPM>, ExpectationSemiringPM<LogSemiring,NgramMatchPM,ListPM,MultiListPM,ExpbleuBO>> 
{

	ExpbleuBO pBO = new ExpbleuBO();  //USED TO COMPUTE t=r x s

	private List<FeatureTemplate> featureTemplates;
	private HashMap<String,Integer> featureStringToIDTable;
	private String[] refs;
	private double[] featureWeights;
	private HashSet<String> featureSet; 
	private NgramExtractor getNgramHelper;
	private HashMap<HyperEdge, ArrayList<Integer>> edgeAnnotationTbl 
		= new HashMap<HyperEdge,ArrayList<Integer>>();

	public ExpbleuSemiringParser(
			String[] references, 
			List<FeatureTemplate> fts, 
			HashMap<String,Integer> ftbl, 
			double[] weights,
			HashSet<String> fs,
			SymbolTable symtbl)
	{
		super();
		this.refs = references;
		this.featureStringToIDTable = ftbl;
		this.featureTemplates = fts;
		this.featureWeights = weights;
		this.featureSet = fs;
		this.getNgramHelper = new NgramExtractor(symtbl, MRConfig.ngramStateID,false,MRConfig.baselineLMOrder);
	}
	
	//CREATE EDGE WEIGHT s=\Delta_Pe, t=\Delta P_e * m_e
	@Override
	protected ExpectationSemiringPM<LogSemiring,NgramMatchPM,ListPM,MultiListPM,ExpbleuBO> createNewXWeight() 
	{
		ListPM s = new ListPM();  //TO SAVE s=\Delta_Pe
		MultiListPM t = new MultiListPM();  //TO SAVE t=me*\Delta_Pe
		return new ExpectationSemiringPM<LogSemiring,NgramMatchPM,ListPM,MultiListPM,ExpbleuBO>(s,t,pBO);
	}

	//COMPUTES X IN E_K,X FOR THIS EDGE
	@Override
	protected ExpectationSemiringPM<LogSemiring,NgramMatchPM,ListPM,MultiListPM,ExpbleuBO> getEdgeXWeight(HyperEdge dt, HGNode parentItem) 
	{
		// s = \Delta P_e = f_e * p_e (p8 of the paper)
		HashMap<Integer,SignedValue> dpe= new HashMap<Integer,SignedValue>();

		//THIS IS TO COMPUTE exp(\sum f_i \theta_i) (IN log FORM) WHICH IS THE (UNNORMALIZED) PROBABILITY P_e
		double logProb = 0;
		for(FeatureTemplate ft : this.featureTemplates)
		{  //FOR ALL TYPES OF FEATURES
			
			HashMap<String,Double> firedfeatures = new HashMap<String,Double>();
			ft.getFeatureCounts(dt, firedfeatures, this.featureSet, 1.0);
						
			for( Map.Entry<String, Double> feature : firedfeatures.entrySet())
			{  //FOR EACH FEATURE OF THIS TYPE	
				Integer featID = this.featureStringToIDTable.get(feature.getKey());
				if(dpe.containsKey(featID)){
					dpe.get(featID).add(SignedValue.createSignedValueFromRealNumber(feature.getValue()));
				}
				else{
					dpe.put(featID, SignedValue.createSignedValueFromRealNumber(feature.getValue()));  //SAVE FEATURE VALUE TO THIS EDGE
				}
				logProb += this.featureWeights[featID] * feature.getValue();
			}
		}
		
		//COMPUTE DERIVATIVE FOR EACH FEATURE:
		
		for(Integer featID : dpe.keySet())
		{
			//if( dt.getRule() == null )
			//	dpe.get(featID).setToZero();  //????? (AT THE GOAL EDGE)
			//else
				dpe.get(featID).multiLogNumber(logProb);  //fe*pe, (A VECTOR)
		}

		//SAVE THE DERIVATIVES
		ListPM deltaPe= new ListPM(new SparseMap(dpe));
		
		//------------ FINISHED S COMPUTING ---------------
		
		//?????
		//AT GOAL EDGE, p_e = 1, Delta_pe=0
		if(dt.getRule() == null)
		{
			//goal node edge , just return zero
			MultiListPM te = new MultiListPM();
			te.setToZero();
			return new ExpectationSemiringPM<LogSemiring,NgramMatchPM,ListPM,MultiListPM,ExpbleuBO>(deltaPe, te, this.pBO);
		}
		
		//FOR TESTING
//		this.getNgramHelper.getTransitionNgrams(dt, 1, 4);
//		
//		System.out.println("Pause......");
//		try 
//		{
//			Thread.sleep(200000);
//		} 
//		catch(InterruptedException e) 
//		{
//		}
		
		
		// t = \Delta P_e * r_e  (NAMELY \Delta m_e*P_e^T)
		
		//FIRST COMPUTE m_e
		ArrayList<Integer> edgeAnnotation;
		if(!edgeAnnotationTbl.containsKey(dt))  //IF THIS EDGE HAS NOT BE PROCESSED
		{
			edgeAnnotation = new ArrayList<Integer>(5);  //SAVE N-GRAM MATCH FOR THIS EDGE
			int bleuOrder = 4;
			
			//COMPUTE NGRAM MATCH ON EACH EDGE
			//0: HYPOTHESIS LENGTH, JUST GIVE A FAKE NUMBER HERE, NOT USED
			//this.getNgramHelper.getTransitionNgrams: BUILD THE HYPOTHESIS N-GRAM TABLE FROM HYPEREDGE
			//BLEU.constructMaxRefCountTable: BUILD THE REFERENCE N-GRAM TABLE
			//bleuOrder: 4
			int[] ngramMatchesOnEdge = BLEU.computeNgramMatches(
					0, 
					this.getNgramHelper.getTransitionNgrams(dt, 1, bleuOrder), 
					BLEU.constructMaxRefCountTable(this.refs,bleuOrder), 
					bleuOrder);
			
			for(int i = 1; i <= 4; ++i)
			{
				edgeAnnotation.add(ngramMatchesOnEdge[i]);
			}
			
			//COMPUTE THE SENTENCE LENGTH FOR THIS EDGE
			int numOfTerms = dt.getRule().getEnglish().length - dt.getRule().getArity();
			edgeAnnotation.add(numOfTerms);
			edgeAnnotationTbl.put(dt, edgeAnnotation);
		}
		else
		{
			edgeAnnotation = edgeAnnotationTbl.get(dt);
		}
		
		//SAVE THE me VALUES
		SignedValue[] me = new SignedValue[5];
		for(int i = 0; i < 5; ++i){
			me[i] = SignedValue.createSignedValueFromRealNumber(edgeAnnotation.get(i));
		}
		
		//COMPUTE te = me x \Delta P_e^T
		NgramMatchPM mePM = new NgramMatchPM(me);
		MultiListPM te = pBO.bilinearMulti(mePM, deltaPe);
		
		//X = S x T IS ALSO A P-MODULE, THEREFORE NEED TO RETURN A PM  
		return new ExpectationSemiringPM<LogSemiring,NgramMatchPM,ListPM,MultiListPM,ExpbleuBO>(deltaPe,te,this.pBO);  //S, T, BO
	}

	//NOT USED
	@Override
	public void normalizeGoal() {
		// TODO Auto-generated method stub

	}

	//CREATES EXPECTATION SEMIRING FOR p & r(NGRAM MATCH),
	@Override 
	protected ExpectationSemiring<LogSemiring,NgramMatchPM> createNewKWeight() 
	{
		LogSemiring p = new LogSemiring();
		NgramMatchPM r= new NgramMatchPM();

		return new ExpectationSemiring<LogSemiring,NgramMatchPM>(p,r);
	}

	//COMPUTES K IN E_K,X FOR THIS EDGE
	@Override 
	protected ExpectationSemiring<LogSemiring,NgramMatchPM> getEdgeKWeight(HyperEdge dt, HGNode parentItem) 
	{
		// p_e
		double logTransitionProb = 0;
		
		//THE SAME AS IN getEdgeXWeight
		for(FeatureTemplate ft : featureTemplates){
			HashMap<String,Double> firedfeatures = new HashMap<String,Double>();

			ft.getFeatureCounts(dt, firedfeatures, featureSet, 1.0);
			for(Map.Entry<String, Double> feature : firedfeatures.entrySet()){
				logTransitionProb += this.featureWeights[this.featureStringToIDTable.get(feature.getKey())] * feature.getValue();
			}
		}
//		System.out.println(logTransitionProb);
		
		LogSemiring pe = new  LogSemiring(logTransitionProb);
		
		if(dt.getRule() == null)  //GOAL EDGE, log P_e = 0
		{
			NgramMatchPM re = new NgramMatchPM();
			
			//pe.setToZero();  //??????
			
			re.setToZero();
			return new ExpectationSemiring<LogSemiring,NgramMatchPM>(pe,re);
		}
		else{
			// r_e =  p_e * m_e
			ArrayList<Integer> edgeAnnotation;
			if(!edgeAnnotationTbl.containsKey(dt)){
				int bleuOrder = 4;
				//			System.out.println(this.symtbl.getWords(dt.getRule().getEnglish()));

				Map<String,Integer> edgeNgramTbl  = this.getNgramHelper.getTransitionNgrams(dt, 1, bleuOrder);
				//			for(Map.Entry<String, Integer> ent : edgeNgramTbl.entrySet()){
				//				System.out.print(ent.getKey());
				//				System.out.print(ent.getValue() + " ");
				//			}
				//			System.out.println();
				
				//EXTRACT NGRAM MATCHES
				int[] ngramMatchesOnEdge = BLEU.computeNgramMatches(0, 
						edgeNgramTbl,
						BLEU.constructMaxRefCountTable(refs,bleuOrder),
						bleuOrder);
				edgeAnnotation = new ArrayList<Integer>(5);
				for(int i = 1; i <=4; ++i){
					edgeAnnotation.add(ngramMatchesOnEdge[i]);
				}
				
				int numOfTerms = dt.getRule().getEnglish().length - dt.getRule().getArity();
				edgeAnnotation.add(numOfTerms);
				edgeAnnotationTbl.put(dt, edgeAnnotation);
			}
			else{
				edgeAnnotation = edgeAnnotationTbl.get(dt);
			}
			
			SignedValue[] peme = new SignedValue[5];
			for(int i = 0; i < 5; ++i){
//				System.out.println(i + " " + ngramMatchesOnEdge[i]);
				peme[i] = SignedValue.createSignedValueFromRealNumber(edgeAnnotation.get(i));
				peme[i].multiLogNumber(pe.getLogValue());
//				peme[i-1].printInfor();
			}
			NgramMatchPM re = new NgramMatchPM(peme);

			//RETURNS E_P,R
			return new ExpectationSemiring<LogSemiring,NgramMatchPM>(pe,re);
		}
	}
	
	// added functions 
	public void parseOverHG(){
		this.clearState();
		this.insideEstimationOverHG();
		this.outsideEstimationOverHG();
//		ExpectationSemiring<LogSemiring,NgramMatchPM> goalK = this.getGoalK();
//		System.out.println("Z = " + goalK.getP().getRealValue());
//		System.out.println("1gram Match Expectation = ");
//		goalK.getR().getNgramMatchExp()[0].printInfor();
//		System.out.println("2gram Match Expectation = ");
//		goalK.getR().getNgramMatchExp()[1].printInfor();
//		System.out.println("3gram Match Expectation = ");
//		goalK.getR().getNgramMatchExp()[2].printInfor();
//		System.out.println("4gram Match Expectation = ");
//		goalK.getR().getNgramMatchExp()[3].printInfor();
//		ExpectationSemiringPM<LogSemiring,NgramMatchPM,ListPM,MultiListPM,ExpbleuBO> goalX = this.getGoalX();
//		for(int id : goalX.getS().getValue().getIds()){
//			System.out.println("d(Z)/dtheta " + id + " = ");
//			goalX.getS().getValue().getValueAt(id).printInfor();
//		}
//		for(int i = 0; i < 4; ++i){
//			for(int id : goalX.getT().getListPMs()[i].getValue().getIds()){
//				System.out.println("dm " + i + " at " + id + " = ");
//				goalX.getT().getListPMs()[i].getValue().getValueAt(id).printInfor();
//			}
//		}
	}
	
	public double [] getNgramMatches(){
		// m(i) = m'(i)/Z (BECAUSE p_e WAS NOT NORMALIZED)
		// note here, only 4grams matches are returned, the 5th value is the length expectation
		double [] ngramMatch = new double[5];
		double logZ = this.getGoalK().getP().getLogValue();  //GET THE NORMALIZATION TERM
		for(int i = 0; i < 5; ++i){
			SignedValue normalizedNgramMatch = this.getGoalK().getR().getNgramMatchExp()[i].duplicate();
			normalizedNgramMatch.multiLogNumber(-logZ);
			ngramMatch[i] = normalizedNgramMatch.convertToRealValue();
		}
		return ngramMatch;
	}
	
	public double [] getGradients(int ngram){
		// d m(i) = d(m'(i)/Z) = dm'(i)Z - m'(i)dZ/ Z^2  ?????
		// d m(i) = d(m'(i)/Z) = ( dm'(i)Z - m'(i)dZ )/ Z^2
		//note here, when ngram == 4, the returned value is the gradients of the length expactation function. 
		int numOfFeats = this.featureWeights.length;
		double [] gradients = new double[numOfFeats];
		double logZ = this.getGoalK().getP().getLogValue();
		SignedValue m = this.getGoalK().getR().getNgramMatchExp()[ngram];
		ListPM dm = this.getGoalX().getT().getListPMs()[ngram];
		ListPM dz = this.getGoalX().getS();
		
		for(int id : dm.getValue().getIds())
		{
			SignedValue mdz = SignedValue.multi(m, dz.getValue().getValueAt(id));
			SignedValue dmz = dm.getValue().getValueAt(id).duplicate();
			dmz.multiLogNumber(logZ);
			mdz.negate();
			dmz.add(mdz);
			dmz.multiLogNumber(-2*logZ);
			gradients[id] = dmz.convertToRealValue();
		}
		return gradients;
	}
}
