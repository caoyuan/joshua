/* ----------------------------------------------------------------------------
 * This file was automatically generated by SWIG (http://www.swig.org).
 * Version 1.3.40
 *
 * Do not make changes to this file unless you know what you are doing--modify
 * the SWIG interface file instead.
 * ----------------------------------------------------------------------------- */

package joshua.decoder.ff.lm.srilm;

public class srilm {
  public static SWIGTYPE_p_unsigned_int new_unsigned_array(int nelements) {
    long cPtr = srilmJNI.new_unsigned_array(nelements);
    return (cPtr == 0) ? null : new SWIGTYPE_p_unsigned_int(cPtr, false);
  }

  public static void delete_unsigned_array(SWIGTYPE_p_unsigned_int ary) {
    srilmJNI.delete_unsigned_array(SWIGTYPE_p_unsigned_int.getCPtr(ary));
  }

  public static long unsigned_array_getitem(SWIGTYPE_p_unsigned_int ary, int index) {
    return srilmJNI.unsigned_array_getitem(SWIGTYPE_p_unsigned_int.getCPtr(ary), index);
  }

  public static void unsigned_array_setitem(SWIGTYPE_p_unsigned_int ary, int index, long value) {
    srilmJNI.unsigned_array_setitem(SWIGTYPE_p_unsigned_int.getCPtr(ary), index, value);
  }

  public static SWIGTYPE_p_Ngram initLM(int order, int start_id, int end_id) {
    long cPtr = srilmJNI.initLM(order, start_id, end_id);
    return (cPtr == 0) ? null : new SWIGTYPE_p_Ngram(cPtr, false);
  }

  public static SWIGTYPE_p_Vocab initVocab(int start, int end) {
    long cPtr = srilmJNI.initVocab(start, end);
    return (cPtr == 0) ? null : new SWIGTYPE_p_Vocab(cPtr, false);
  }

  public static long getIndexForWord(String s) {
    return srilmJNI.getIndexForWord(s);
  }

  public static String getWordForIndex(long i) {
    return srilmJNI.getWordForIndex(i);
  }

  public static int readLM(SWIGTYPE_p_Ngram ngram, String filename) {
    return srilmJNI.readLM(SWIGTYPE_p_Ngram.getCPtr(ngram), filename);
  }

  public static float getWordProb(SWIGTYPE_p_Ngram ngram, long word, SWIGTYPE_p_unsigned_int context) {
    return srilmJNI.getWordProb(SWIGTYPE_p_Ngram.getCPtr(ngram), word, SWIGTYPE_p_unsigned_int.getCPtr(context));
  }

  public static float getProb_lzf(SWIGTYPE_p_Ngram ngram, SWIGTYPE_p_unsigned_int context, int hist_size, long cur_wrd) {
    return srilmJNI.getProb_lzf(SWIGTYPE_p_Ngram.getCPtr(ngram), SWIGTYPE_p_unsigned_int.getCPtr(context), hist_size, cur_wrd);
  }

  public static long getBOW_depth(SWIGTYPE_p_Ngram ngram, SWIGTYPE_p_unsigned_int context, int hist_size) {
    return srilmJNI.getBOW_depth(SWIGTYPE_p_Ngram.getCPtr(ngram), SWIGTYPE_p_unsigned_int.getCPtr(context), hist_size);
  }

  public static float get_backoff_weight_sum(SWIGTYPE_p_Ngram ngram, SWIGTYPE_p_unsigned_int context, int hist_size, int min_len) {
    return srilmJNI.get_backoff_weight_sum(SWIGTYPE_p_Ngram.getCPtr(ngram), SWIGTYPE_p_unsigned_int.getCPtr(context), hist_size, min_len);
  }

  public static int getVocab_None() {
    return srilmJNI.getVocab_None();
  }

  public static void write_vocab_map(SWIGTYPE_p_Vocab vo, String fname) {
    srilmJNI.write_vocab_map(SWIGTYPE_p_Vocab.getCPtr(vo), fname);
  }

  public static void write_default_vocab_map(String fname) {
    srilmJNI.write_default_vocab_map(fname);
  }

  public static String getWordForIndex_Vocab(SWIGTYPE_p_Vocab vo, long i) {
    return srilmJNI.getWordForIndex_Vocab(SWIGTYPE_p_Vocab.getCPtr(vo), i);
  }

  public static long getIndexForWord_Vocab(SWIGTYPE_p_Vocab vo, String s) {
    return srilmJNI.getIndexForWord_Vocab(SWIGTYPE_p_Vocab.getCPtr(vo), s);
  }

}
