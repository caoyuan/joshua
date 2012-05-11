package joshua.pro.classifier.maxent.edu.stanford.nlp.classify;

import joshua.pro.classifier.maxent.edu.stanford.nlp.ling.Datum;
import joshua.pro.classifier.maxent.edu.stanford.nlp.stats.Counter;

public interface ProbabilisticClassifier<L, F> extends Classifier<L, F>
{
  public Counter<L> probabilityOf(Datum<L, F> example);
  public Counter<L> logProbabilityOf(Datum<L, F> example);
}
