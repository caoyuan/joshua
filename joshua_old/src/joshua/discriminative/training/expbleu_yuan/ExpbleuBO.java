package joshua.discriminative.training.expbleu_yuan;

import joshua.discriminative.semiring_parsingv2.SignedValue;
import joshua.discriminative.semiring_parsingv2.bilinear_operator.BilinearOperator;
import joshua.discriminative.semiring_parsingv2.pmodule.ListPM;
import joshua.discriminative.semiring_parsingv2.pmodule.SparseMap;

public class ExpbleuBO implements BilinearOperator<NgramMatchPM, ListPM, MultiListPM> {

	//DEFINE OPERATION R X S -> T WHICH IS BILINEAR
	public MultiListPM bilinearMulti(NgramMatchPM r, ListPM s) {
		ListPM[] product = new ListPM[5];
		
		//COMPUTE THE MATRIX r x s^T
		for(int i = 0; i < 5; ++i)
		{  //FOR EACH ENTRY IN r (1-4 NGRAM MATCH)
			SparseMap vectorTimesSigned = s.getValue().duplicate();
			
			for(SignedValue v : vectorTimesSigned.getValues())
			{  //FOR EACH ENTRY IN s (dp/d theta_i)
				v.multi(r.getNgramMatchExp()[i]);
			}
			
			product[i] = new ListPM(vectorTimesSigned);
		}
		return new MultiListPM(product);
	}
}
