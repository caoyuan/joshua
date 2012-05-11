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
package joshua.decoder.hypergraph;

/**
* @author Zhifei Li, <zhifei.work@gmail.com>
* @version $LastChangedDate: 2010-01-14 22:11:52 -0500 (Thu, 14 Jan 2010) $
*/

public class TrivialInsideOutside extends DefaultInsideOutside {
//	used by inside-outside estimation
	protected  double getHyperedgeLogProb(HyperEdge dt, HGNode parent_it){
		return dt.getTransitionLogP(false);//TODO this is very bad in terms of computation
	}
}
