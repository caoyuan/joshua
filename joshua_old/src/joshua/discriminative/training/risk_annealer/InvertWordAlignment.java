package joshua.discriminative.training.risk_annealer;

import java.io.BufferedWriter;
import java.io.IOException;

import joshua.util.FileUtility;
import joshua.util.io.LineReader;

public class InvertWordAlignment {

	public static String invert(String input) {
		// Example input: 0-0 3-1 3-2 3-3 3-4 3-5 2-6 3-7
		StringBuffer sb = new StringBuffer();
		String[] pairs = input.split("\\s+");
		for (int i = 0; i < pairs.length; ++i) {
			String[] ids = pairs[i].split("\\-");
			sb.append(ids[1]);
			sb.append("-");
			sb.append(ids[0]);
			if (i < pairs.length -1)
				sb.append(" ");
		}
		return sb.toString();
	}
	
	public static void main(String[] args) {
		if(args.length != 2){
			System.out.println("Wrong number of parameters!");
			System.exit(1);
		}
	
		String alingmentInputFile = args[0].trim();
		String alingmentOutputFile = args[1].trim();
		try {		
			LineReader reader = new LineReader(alingmentInputFile);
			BufferedWriter output = FileUtility.getWriteFileStream(alingmentOutputFile);
			for (String example : reader) {
				output.write( InvertWordAlignment.invert(example)  + "\n" );
			}
			reader.close();
			output.flush();
			output.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
}
