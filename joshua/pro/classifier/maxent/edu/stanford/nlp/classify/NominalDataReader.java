package joshua.pro.classifier.maxent.edu.stanford.nlp.classify;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Set;
import java.util.StringTokenizer;
import java.io.File;

import joshua.pro.classifier.maxent.edu.stanford.nlp.ling.RVFDatum;
import joshua.pro.classifier.maxent.edu.stanford.nlp.stats.ClassicCounter;
import joshua.pro.classifier.maxent.edu.stanford.nlp.objectbank.ObjectBank;
import joshua.pro.classifier.maxent.edu.stanford.nlp.util.HashIndex;
import joshua.pro.classifier.maxent.edu.stanford.nlp.util.Index;

/**
 * @author Kristina Toutanova
 *         Sep 14, 2004
 *         A class to read some UCI datasets into RVFDatum. Willl incrementally add formats
 *
 * Made type-safe by Sarah Spikes (sdspikes@cs.stanford.edu)
 */
public class NominalDataReader {
  HashMap<String, Index<String>> indices = new HashMap<String, Index<String>>(); // an Index for each feature so that its values are coded as integers

  /**
   * the class is the last column and it skips the next-to-last column because it is a unique id in the audiology data
   *
   */
  RVFDatum<String, Integer> readDatum(String line, String separator, HashMap<Integer, Index<String>> indices) {
    StringTokenizer st = new StringTokenizer(line, separator);
    //int fno = 0;
    ArrayList<String> tokens = new ArrayList<String>();
    while (st.hasMoreTokens()) {
      String token = st.nextToken();
      tokens.add(token);
    }
    String[] arr = (String[])tokens.toArray();
    Set<Integer> skip = new HashSet<Integer>();
    skip.add(Integer.valueOf(arr.length - 2));
    return readDatum(arr, arr.length - 1, skip, indices);
  }

  RVFDatum<String, Integer> readDatum(String[] values, int classColumn, Set<Integer> skip, HashMap<Integer, Index<String>> indices) {
    ClassicCounter<Integer> c = new ClassicCounter<Integer>();
    RVFDatum<String, Integer> d = new RVFDatum<String, Integer>(c);
    int attrNo = 0;
    for (int index = 0; index < values.length; index++) {
      if (index == classColumn) {
        d.setLabel(values[index]);
        continue;
      }
      if (skip.contains(Integer.valueOf(index))) {
        continue;
      }
      Integer featKey = Integer.valueOf(attrNo);
      Index<String> ind = indices.get(featKey);
      if (ind == null) {
        ind = new HashIndex<String>();
        indices.put(featKey, ind);
      }
      // MG: condition on isLocked is useless, since add(E) contains such a condition:
      //if (!ind.isLocked()) {
        ind.add(values[index]);
      //}
      int valInd = ind.indexOf(values[index]);
      if (valInd == -1) {
        valInd = 0;
        System.err.println("unknown attribute value " + values[index] + " of attribute " + attrNo);
      }
      c.incrementCount(featKey, valInd);
      attrNo++;

    }
    return d;
  }

  /**
   * Read the data as a list of RVFDatum objects. For the test set we must reuse the indices from the training set
   *
   */
  ArrayList<RVFDatum<String, Integer>> readData(String filename, HashMap<Integer, Index<String>> indices) {
    try {

      String sep = ", ";
      ArrayList<RVFDatum<String, Integer>> examples = new ArrayList<RVFDatum<String, Integer>>();
      for(String line : ObjectBank.getLineIterator(new File(filename))) {
        RVFDatum<String, Integer> next = readDatum(line, sep, indices);
        examples.add(next);
      }
      return examples;
    } catch (Exception e) {
      e.printStackTrace();
    }
    return null;
  }


}
