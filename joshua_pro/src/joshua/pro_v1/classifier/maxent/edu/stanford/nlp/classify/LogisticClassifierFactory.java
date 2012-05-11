package joshua.pro.classifier.maxent.edu.stanford.nlp.classify;

import java.util.List;

import joshua.pro.classifier.maxent.edu.stanford.nlp.ling.RVFDatum;
import joshua.pro.classifier.maxent.edu.stanford.nlp.optimization.DiffFunction;
import joshua.pro.classifier.maxent.edu.stanford.nlp.optimization.Minimizer;
import joshua.pro.classifier.maxent.edu.stanford.nlp.optimization.QNMinimizer;
import joshua.pro.classifier.maxent.edu.stanford.nlp.util.ErasureUtils;
import joshua.pro.classifier.maxent.edu.stanford.nlp.util.Index;
import joshua.pro.classifier.maxent.edu.stanford.nlp.util.ReflectionLoading;


public class LogisticClassifierFactory<L,F> implements ClassifierFactory<L, F, LogisticClassifier<L,F>> {
  /**
   *
   */
  private static final long serialVersionUID = 1L;
  private double[] weights;
  private Index<F> featureIndex;
  private L[] classes = ErasureUtils.<L>mkTArray(Object.class,2);


  public LogisticClassifier<L,F> trainWeightedData(GeneralDataset<L,F> data, float[] dataWeights){
    if(data instanceof RVFDataset)
      ((RVFDataset<L,F>)data).ensureRealValues();
    if (data.labelIndex.size() != 2) {
      throw new RuntimeException("LogisticClassifier is only for binary classification!");
    }

    Minimizer<DiffFunction> minim;
      LogisticObjectiveFunction lof = null;
      if(data instanceof Dataset<?,?>)
        lof = new LogisticObjectiveFunction(data.numFeatureTypes(), data.getDataArray(), data.getLabelsArray(), new LogPrior(LogPrior.LogPriorType.QUADRATIC),dataWeights);
      else if(data instanceof RVFDataset<?,?>)
        lof = new LogisticObjectiveFunction(data.numFeatureTypes(), data.getDataArray(), data.getValuesArray(), data.getLabelsArray(), new LogPrior(LogPrior.LogPriorType.QUADRATIC),dataWeights);
      minim = new QNMinimizer(lof);
      weights = minim.minimize(lof, 1e-4, new double[data.numFeatureTypes()]);

    featureIndex = data.featureIndex;
    classes[0] = data.labelIndex.get(0);
    classes[1] = data.labelIndex.get(1);
    return new LogisticClassifier<L,F>(weights,featureIndex,classes);
  }

  public LogisticClassifier<L,F> trainClassifier(GeneralDataset<L, F> data) {
    return trainClassifier(data, 0.0);
  }

  public LogisticClassifier<L,F> trainClassifier(GeneralDataset<L, F> data, LogPrior prior, boolean biased) {
    return trainClassifier(data, 0.0, 1e-4, prior, biased);
  }

  public LogisticClassifier<L,F> trainClassifier(GeneralDataset<L, F> data, double l1reg) {
    return trainClassifier(data, l1reg, 1e-4);
  }

  public LogisticClassifier<L,F> trainClassifier(GeneralDataset<L, F> data, double l1reg, double tol) {
    return trainClassifier(data, l1reg, tol, new LogPrior(LogPrior.LogPriorType.QUADRATIC), false);
  }

  public LogisticClassifier<L,F> trainClassifier(GeneralDataset<L, F> data, double l1reg, double tol, LogPrior prior) {
    return trainClassifier(data, l1reg, tol, prior, false);
  }

  public LogisticClassifier<L,F> trainClassifier(GeneralDataset<L, F> data, double l1reg, double tol, boolean biased) {
    return trainClassifier(data, l1reg, tol, new LogPrior(LogPrior.LogPriorType.QUADRATIC), biased);
  }

  public LogisticClassifier<L,F> trainClassifier(GeneralDataset<L, F> data, double l1reg, double tol, LogPrior prior, boolean biased) {
    if(data instanceof RVFDataset)
      ((RVFDataset<L,F>)data).ensureRealValues();
    if (data.labelIndex.size() != 2) {
      throw new RuntimeException("LogisticClassifier is only for binary classification!");
    }

    Minimizer<DiffFunction> minim;
    if (!biased) {
      LogisticObjectiveFunction lof = null;
      if(data instanceof Dataset<?,?>)
        lof = new LogisticObjectiveFunction(data.numFeatureTypes(), data.getDataArray(), data.getLabelsArray(), prior);
      else if(data instanceof RVFDataset<?,?>)
        lof = new LogisticObjectiveFunction(data.numFeatureTypes(), data.getDataArray(), data.getValuesArray(), data.getLabelsArray(), prior);
      if (l1reg > 0.0) {
        minim = ReflectionLoading.loadByReflection("joshua.pro.classifier.maxent.edu.stanford.nlp.optimization.OWLQNMinimizer", l1reg);
      } else {
        minim = new QNMinimizer(lof);
      }
      weights = minim.minimize(lof, tol, new double[data.numFeatureTypes()]);
    } else {
      BiasedLogisticObjectiveFunction lof = new BiasedLogisticObjectiveFunction(data.numFeatureTypes(), data.getDataArray(), data.getLabelsArray(), prior);
      if (l1reg > 0.0) {
        minim = ReflectionLoading.loadByReflection("joshua.pro.classifier.maxent.edu.stanford.nlp.optimization.OWLQNMinimizer", l1reg);
      } else {
        minim = new QNMinimizer(lof);
      }
      weights = minim.minimize(lof, tol, new double[data.numFeatureTypes()]);
    }

    featureIndex = data.featureIndex;
    classes[0] = data.labelIndex.get(0);
    classes[1] = data.labelIndex.get(1);
    return new LogisticClassifier<L,F>(weights,featureIndex,classes);
  }

  @Deprecated //this method no longer required by the ClassifierFactory Interface.
  public LogisticClassifier<L, F> trainClassifier(List<RVFDatum<L, F>> examples) {
    // TODO Auto-generated method stub
    return null;
  }

}
