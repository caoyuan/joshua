/* This file is part of the Joshua Machine Translation System.
 * 
 * Joshua is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation; either version 2.1
 * of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the Free
 * Software Foundation, Inc., 59 Temple Place, Suite 330, Boston,
 * MA 02111-1307 USA
 */
package joshua.prefix_tree;

import joshua.corpus.suffix_array.Pattern;

/**
 * Represents a tuple used during prefix tree construction.
 *
 * @author Lane Schwartz
 * @see Lopez (2008) PhD Thesis, Algorithm 2, p 76
 * @version $LastChangedDate: 2009-05-23 00:31:12 -0400 (Sat, 23 May 2009) $
 */
class Tuple {

	/**
	 * Pattern corresponding to the prefix node (NOT the pattern
	 * corresponding to the new node that will be constructed).
	 */
	final Pattern pattern;
	
	/**
	 * Start index of the pattern in the source input sentence
	 * (inclusive, 1-based).
	 */
	final int spanStart;
	
	/**
	 * End index of the pattern in the source input sentence
	 * (inclusive, 1-based).
	 */
	final int spanEnd;
	
	/**
	 * Node in the prefix tree to which a new node (corresponding
	 * to the pattern) will be attached.
	 */
	final Node prefixNode;

	/**
	 * Constructs a new tuple.
	 * 
	 * @param pattern Pattern corresponding to the prefix node
	 *            (NOT the pattern corresponding to the new
	 *            node that will be constructed).
	 * @param spanStart Start index of the pattern in the source
	 *            input sentence (inclusive, 1-based).
	 * @param spanEnd End index of the pattern in the source
	 *            input sentence (inclusive, 1-based).
	 * @param prefixNode Node in the prefix tree to which a new
	 *            node (corresponding to the pattern) will be
	 *            attached.
	 */
	Tuple(Pattern pattern, int spanStart, int spanEnd, Node prefixNode) {
		this.pattern = pattern;
		this.spanStart = spanStart;
		this.spanEnd = spanEnd;
		this.prefixNode = prefixNode;
	}

}
