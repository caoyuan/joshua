package joshua.pro.classifier.maxent.edu.stanford.nlp.optimization;

/**
 * Indicates that a function has a method for supplying an intitial value.
 *
 * @author <a href="mailto:klein@cs.stanford.edu">Dan Klein</a>
 * @version 1.0
 * @since 1.0
 */
public interface HasInitial {
  /**
   * Returns the intitial point in the domain (but not necessarily a feasible one).
   *
   * @return a domain point
   */
  double[] initial();
}
