/*
 * This file is part of the Joshua Machine Translation System.
 * 
 * Joshua is free software; you can redistribute it and/or modify it under the terms of the GNU
 * Lesser General Public License as published by the Free Software Foundation; either version 2.1 of
 * the License, or (at your option) any later version.
 * 
 * This library is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without
 * even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * Lesser General Public License for more details.
 * 
 * You should have received a copy of the GNU Lesser General Public License along with this library;
 * if not, write to the Free Software Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA
 * 02111-1307 USA
 */

package joshua.pro;

import java.io.IOException;
import java.util.Vector;

import joshua.pro.classifier.maxent.edu.stanford.nlp.classify.ColumnDataClassifier;

public class ClassifierMaxEnt implements ClassifierInterface {
  @Override
  public double[] runClassifier(Vector<String> samples, double[] initialLambda, int featDim) {
    System.out.println("--------- Start MaxEnt training ----------");

    double[] lambda = new double[featDim + 1];
    // propFilePath =
    // "/home/yuan/Desktop/stanford-classifier-2011-09-16/src/examples/iris2007.prop";

    try {
      lambda = ColumnDataClassifier.run(propFilePath, samples, featDim);
    } catch (IOException e) {
      e.printStackTrace();
    }

    System.out.println("--------- End MaxEnt training ----------");

    /*
     * try { Thread.sleep(20000); } catch(InterruptedException e) { }
     */

    return lambda;
  }

  @Override
  /*
   * for Stanford MaxEnt tool: param[0] = prop file path
   */
  public void setClassifierParam(String[] param) {
    if (param == null) {
      System.out.println("ERROR: must provide parameters for Stanford Max-Entropy classifier!");
      System.exit(10);
    } else
      propFilePath = param[0];
  }

  String propFilePath;
}
