/*
 * METEORNEXT metric support for ZMERT
 */

package joshua.zmert;

import java.io.File;
import java.net.MalformedURLException;
import java.util.ArrayList;
import java.util.StringTokenizer;

import joshua.zmert.meteor.src.edu.cmu.meteor.scorer.MeteorConfiguration;
import joshua.zmert.meteor.src.edu.cmu.meteor.scorer.MeteorScorer;
import joshua.zmert.meteor.src.edu.cmu.meteor.scorer.MeteorStats;
import joshua.zmert.meteor.src.edu.cmu.meteor.util.Constants;

/*
 * Default: en lowercase '0.5 1.0 0.5 0.5' 'exact stem synonym paraphrase' '1.0 0.5 0.5 0.5'
 * 
 * Parameters:
 * 1. Language = (en | cz | de | es | fr | other)
 * 2. Normalization = (none | lowercase | punct | nopunct)
 * 3. Parameters (alpha beta gamma delta) = '0.5 1.0 0.5 0.5'
 * 4. Modules = 'exact stem synonym paraphrase'
 * 5. Module weights = '1.0 0.5 0.5 0.5'
 * 
 * Normalization:
 * none - use raw decoder output
 * lowercase - only lowercase (recommended)
 * punct - lowercase, tokenize punctuation
 * nopunct - lowercase, strip punctuation
 * 
 */

public class Meteor extends EvaluationMetric {

	private MeteorConfiguration config = new MeteorConfiguration();
	private MeteorScorer scorer;

	// Use configuration as-is
	public Meteor() {
		initialize();
	}

	// Update configuration with options, then initialize
	public Meteor(String[] m_args) {

		// Meteor Options

		// Language
		config.setLanguage(m_args[0]);

		// Normalization
		if (m_args[1].equals("none"))
			config.setNormalization(Constants.NO_NORMALIZE);
		else if (m_args[1].equals("lowercase"))
			config.setNormalization(Constants.NORMALIZE_LC_ONLY);
		else if (m_args[1].equals("punct"))
			config.setNormalization(Constants.NORMALIZE_KEEP_PUNCT);
		else if (m_args[1].equals("nopunct"))
			config.setNormalization(Constants.NORMALIZE_NO_PUNCT);

		// Parameters in form '0.5 1.0 0.5 0.5'
		ArrayList<Double> params = new ArrayList<Double>();
		StringTokenizer tok = new StringTokenizer(m_args[2]);
		while (tok.hasMoreTokens())
			params.add(Double.parseDouble(tok.nextToken()));
		config.setParameters(params);

		// Modules in form 'exact stem synonym paraphrase'
		ArrayList<Integer> modules = new ArrayList<Integer>();
		tok = new StringTokenizer(m_args[3]);
		while (tok.hasMoreTokens())
			modules.add(Constants.getModuleID(tok.nextToken()));
		config.setModules(modules);

		// Module weights in form '1.0 0.5 0.5 0.5'
		ArrayList<Double> weights = new ArrayList<Double>();
		tok = new StringTokenizer(m_args[4]);
		while (tok.hasMoreTokens())
			weights.add(Double.parseDouble(tok.nextToken()));
		config.setModuleWeights(weights);
		
		// Create scorer
		initialize();
	}

	protected void initialize() {
		metricName = "Meteor";
		toBeMinimized = false;
		suffStatsCount = MeteorStats.STATS_LENGTH;
		// Use current config to create scorer
		scorer = new MeteorScorer(config);
	}

	public double bestPossibleScore() {
		return 1.0;
	}

	public double worstPossibleScore() {
		return 0.0;
	}

	public int[] suffStats(String cand_str, int i) {

		// Build ref list
		ArrayList<String> refs = new ArrayList<String>();
		for (int r = 0; r < refsPerSen; r++)
			refs.add(refSentences[i][r]);

		// Score, return as int[] - it really is that simple
		return scorer.getMeteorStats(cand_str, refs).toIntArray();
	}

	public double score(int[] stats) {
		// Convert int[] to MeteorStats object
		MeteorStats statsObject = new MeteorStats(stats);
		// Score
		scorer.computeMetrics(statsObject);
		return statsObject.score;
	}

	public void printDetailedScore_fromStats(int[] stats, boolean oneLiner) {
		// Score stats for verbose output
		MeteorStats statsObject = new MeteorStats(stats);
		scorer.computeMetrics(statsObject);
		if (oneLiner) {
			System.out.println("Meteor: P = "
					+ f4.format(statsObject.precision) + ", R = "
					+ f4.format(statsObject.recall) + ", Frag = "
					+ f4.format(statsObject.fragPenalty) + ", Score = "
					+ f4.format(statsObject.score));
		} else {
			System.out.println("Meteor: P = "
					+ f4.format(statsObject.precision) + ", R = "
					+ f4.format(statsObject.recall) + ", Frag = "
					+ f4.format(statsObject.fragPenalty) + ", Score = "
					+ f4.format(statsObject.score));
		}
	}

}
