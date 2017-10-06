import java.util.Set;
import java.util.Hashtable;
import java.util.ArrayList;
import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;

import Jama.Matrix;

class HMM {
	/* Section for variables regarding the data */
	
	//
	private ArrayList<Sentence> labeled_corpus;
	
	//
	private ArrayList<Sentence> unlabeled_corpus;

	// number of pos tags
	int num_postags;
	
	// mapping POS tags in String to their indices
	Hashtable<String, Integer> pos_tags;
	
	// inverse of pos_tags: mapping POS tag indices to their String format
	Hashtable<Integer, String> inv_pos_tags;
	
	// vocabulary size
	int num_words;

	Hashtable<String, Integer> vocabulary;

	private int max_sentence_length;
	
	/* Section for variables in HMM */
	
	// transition matrix
	private Matrix A;

	// emission matrix
	private Matrix B;

	// prior of pos tags
	private Matrix pi;

	// store the scaled alpha and beta
	private Matrix alpha;
	
	private Matrix beta;

	// scales to prevent alpha and beta from underflowing
	private Matrix scales;

	// logged v for Viterbi
	private Matrix v;
	private Matrix back_pointer;
	private Matrix pred_seq;
	
	// \xi_t(i): expected frequency of pos tag i at position t. Use as an accumulator.
	private Matrix gamma;
	
	// \xi_t(i, j): expected frequency of transiting from pos tag i to j at position t.  Use as an accumulator.
	private Matrix digamma;
	
	// \xi_t(i,w): expected frequency of pos tag i emits word w.
	private Matrix gamma_w;

	// \xi_0(i): expected frequency of pos tag i at position 0.
	private Matrix gamma_0;
	
	/* Section of parameters for running the algorithms */

	// smoothing epsilon for the B matrix (since there are likely to be unseen words in the training corpus)
	// preventing B(j, o) from being 0
	private double smoothing_eps = 0.1;

	// number of iterations of EM
	private int max_iters = 10;
	
	/* Section of variables monitoring training */
	
	// record the changes in log likelihood during EM
	private double[] log_likelihood = new double[max_iters];
	
	/**
	 * Constructor with input corpora.
	 * Set up the basic statistics of the corpora.
	 */
	public HMM(ArrayList<Sentence> _labeled_corpus, ArrayList<Sentence> _unlabeled_corpus) {
        labeled_corpus = _labeled_corpus;
        unlabeled_corpus = _unlabeled_corpus;

        num_postags = 0;
        pos_tags = new Hashtable<>();
        inv_pos_tags = new Hashtable<>();

        num_words = 0;
        vocabulary = new Hashtable<>();

        max_sentence_length = 0;

        for(int i = 0; i < labeled_corpus.size(); ++i){
            Sentence sentence = labeled_corpus.get(i);
            max_sentence_length = Math.max(max_sentence_length, sentence.length());
            for(int j = 0; j < sentence.length(); ++j){
                Word word = sentence.getWordAt(j);

                String lemme = word.getLemme();
                Integer lemme_id = vocabulary.get(lemme);
                if(lemme_id == null){
                    vocabulary.put(lemme, num_words);
                    ++num_words;
                }

                String pos_tag = word.getPosTag();
                Integer tag_id = pos_tags.get(pos_tag);
                if(tag_id == null){
                    pos_tags.put(pos_tag, num_postags);
                    inv_pos_tags.put(num_postags, pos_tag);
                    ++num_postags;
                }
            }
        }

        for(int i = 0; i < unlabeled_corpus.size(); ++i){
            Sentence sentence = unlabeled_corpus.get(i);
            max_sentence_length = Math.max(max_sentence_length, sentence.length());
            for(int j = 0; j < sentence.length(); ++j){
                Word word = sentence.getWordAt(j);

                String lemme = word.getLemme();
                Integer lemme_id = vocabulary.get(lemme);
                if(lemme_id == null){
                    vocabulary.put(lemme, num_words);
                    ++num_words;
                }
            }
        }
	}

	/**
	 * Create HMM variables.
	 */
	public void prepareMatrices() {
       A = new Matrix(new double[num_postags][num_postags]); 
       B = new Matrix(new double[num_postags][num_words]); 
       pi = new Matrix(new double[2][num_postags]); 
	}

	/** 
	 *  MLE A, B and pi on a labeled corpus
	 *  used as initialization of the parameters.
	 */
	public void mle() {
        int[] count_tags = new int[num_postags];
        for(Sentence sentence : labeled_corpus){
            int pre_tag_id = -1;
            for(int i = 0; i < sentence.length(); ++i){
                Word word = sentence.getWordAt(i);
                int tag_id = pos_tags.get(word.getPosTag()).intValue();
                int lemme_id = pos_tags.get(word.getLemme()).intValue();
                count_tags[tag_id]++;
                if(i == 0){
                    pi.set(0, tag_id, pi.get(0, tag_id) + 1);
                }
                else{
                    A.set(pre_tag_id, tag_id, A.get(pre_tag_id, tag_id) + 1);
                }
                B.set(tag_id, lemme_id, B.get(tag_id, lemme_id) + 1);
                pre_tag_id = tag_id;
                if(i == sentence.length() - 1){
                    pi.set(1, tag_id, pi.get(1, tag_id) + 1);
                }
            }
        }

        //normalize
        for(int tag_id = 0; tag_id < num_postags; ++tag_id){
            pi.set(0, tag_id, pi.get(0, tag_id) / labeled_corpus.size());
            pi.set(1, tag_id, pi.get(1, tag_id) / labeled_corpus.size());

            for(int post_tag_id = 0; post_tag_id < num_postags; ++post_tag_id)
                A.set(tag_id, post_tag_id, A.get(tag_id, post_tag_id) / count_tags[tag_id]);

            for(int lemme_id = 0; lemme_id < num_words; ++lemme_id)
                B.set(tag_id, lemme_id, A.get(tag_id, lemme_id) / count_tags[tag_id]);
        }
	}

	/**
	 * Main EM algorithm. 
	 */
	public void em() {
        //Initialize Matrices
        alpha = new Matrix(new double[num_postags][max_sentence_length]); 
        beta = new Matrix(new double[num_postags][max_sentence_length]); 
        scales = new Matrix(new double[2][max_sentence_length]); 

        digamma = new Matrix(new double[num_postags][num_postags]); 

        gamma = new Matrix(new double[1][num_postags]); 
        gamma_0 = new Matrix(new double[1][num_postags]); 
        gamma_w = new Matrix(new double[num_postags][num_words]); 

        for(int iter = 0; iter < 1; ++iter){
            //E-step
            for(Sentence s : unlabeled_corpus){
                expection(s);
            }

            //M-step
            maximization();
        }
	}
	
	/**
	 * Prediction
	 * Find the most likely pos tag for each word of the sentences in the unlabeled corpus.
	 */
	public void predict() {
	}
	
	/**
	 * Output prediction
	 */
	public void outputPredictions(String outFileName) throws IOException {
	}
	
	/**
	 * outputTrainingLog
	 */
	public void outputTrainingLog(String outFileName) throws IOException {
        FileWriter fw = new FileWriter(outFileName);
        BufferedWriter bw = new BufferedWriter(fw);
        for(int i = 0; i < num_postags; ++i){
            for(int j = 0; j < num_postags; ++j)
                bw.write(A.get(i,j) + "\t");
            bw.write("\n");
        }
        bw.close();
        fw.close();
	}
	
	/**
	 * Expection step of the EM (Baum-Welch) algorithm for one sentence.
	 * \xi_t(i,j) and \xi_t(i) are computed for a sentence
	 */
	private double expection(Sentence s) {
        double PO = forward(s);
        backward(s);
        for(int t = 0; t < s.length(); ++t){
            if(t == 0){
                for(int tag_id = 0; tag_id < num_postags; ++tag_id){
                    double new_gamma_0_log = Math.log(alpha.get(tag_id, t)) - Math.log(scales.get(0, t)) + Math.log(beta.get(tag_id, t)) - Math.log(scales.get(1, t)) - PO; 
                    gamma_0.set(0, tag_id, gamma_0.get(0, tag_id) + Math.exp(new_gamma_0_log));
                }
            }

            int lemme_id = vocabulary.get(s.getWordAt(t).getLemme()).intValue();
            for(int tag_id = 0; tag_id < num_postags; ++tag_id){
                double new_gamma_w_log = Math.log(alpha.get(tag_id, t)) - Math.log(scales.get(0, t)) + Math.log(beta.get(tag_id, t)) - Math.log(scales.get(1, t)) - PO; 
                gamma_w.set(tag_id, lemme_id, gamma_w.get(0, tag_id) + Math.exp(new_gamma_w_log));
                gamma.set(0, tag_id, gamma.get(0, tag_id) + Math.exp(new_gamma_w_log));
            }

            if(t < s.length() - 1){
                for(int tag_id = 0; tag_id < num_postags; ++tag_id){
                    int post_lemme_id = vocabulary.get(s.getWordAt(t+1).getLemme()).intValue();
                    for(int post_tag_id = 0; post_tag_id < num_postags; ++post_tag_id){
                        double new_digamma_log = Math.log(alpha.get(tag_id, t)) + Math.log(A.get(tag_id, post_tag_id)) + Math.log(B.get(post_tag_id, post_lemme_id)) + Math.log(beta.get(post_tag_id, t+1)) - PO - Math.log(scales.get(0, t)) - Math.log(scales.get(1, t));
                        digamma.set(tag_id, post_tag_id, digamma.get(tag_id, post_tag_id) + Math.exp(new_digamma_log));
                    }
                }
            }
        }
        return 0;
	}

	/**
	 * Maximization step of the EM (Baum-Welch) algorithm.
	 * Just reestimate A, B and pi using gamma and digamma
	 */
	private void maximization() {
        for(int i = 0; i < num_postags; ++i){
            for(int j = 0; j < num_postags; ++j){
                A.set(i, j, digamma.get(i, j) / gamma.get(0, i));
            }

            for(int j = 0; j < num_words; ++j){
                B.set(i, j, gamma_w.get(i, j) / gamma.get(0, i));
            }
        }

        double scale = 0;
        for(int i = 0; i < num_postags; ++i){
            scale += gamma_0.get(0, i);
        }
        for(int i = 0; i < num_postags; ++i){
            pi.set(0, i, gamma_0.get(0, i) / scale);
        }
	}

	/**
	 * Forward algorithm for one sentence
	 * s: the sentence
	 * alpha: forward probability matrix of shape (num_postags, max_sentence_length)

	 * return: log P(O|\lambda)
	 */
	private double forward(Sentence s) {
        for(int i = 0; i < s.length(); ++i){
            int lemme_id = vocabulary.get(s.getWordAt(i).getLemme()).intValue();
            for(int tag_id = 0; tag_id < num_postags; ++tag_id){
                if(i == 0){
                    alpha.set(tag_id, i, pi.get(0, tag_id) * B.get(tag_id, lemme_id));
                }
                else{
                    double ele = 0;
                    for(int pre_tag_id = 0; pre_tag_id < num_postags; ++ pre_tag_id){
                        ele += alpha.get(pre_tag_id, i-1) * A.get(pre_tag_id, tag_id) * B.get(tag_id, lemme_id);
                    }
                    alpha.set(tag_id, i, ele);
                }
            }
            double scale = 0;
            for(int tag_id = 0; tag_id < num_postags; ++tag_id){
                scale += alpha.get(tag_id, i);
            }
            scales.set(0, i, scale);
            for(int tag_id = 0; tag_id < num_postags; ++tag_id){
                alpha.set(tag_id, i, alpha.get(tag_id, i) / scale);
            }
        }

        double res = 0;
        for(int i = 0; i < s.length(); ++i)
            res += Math.log(scales.get(0, i));
        return res;
	}

	/**
	 * Backward algorithm for one sentence
	 * 
	 * return: log P(O|\lambda)
	 */
	private double backward(Sentence s) {
        for(int i = s.length() - 1; i >= 0; --i){
            int lemme_id = vocabulary.get(s.getWordAt(i).getLemme()).intValue();
            for(int tag_id = 0; tag_id < num_postags; ++tag_id){
                if(i == s.length() - 1){
                    beta.set(tag_id, i, pi.get(1, tag_id) * B.get(tag_id, lemme_id));
                }
                else{
                    double ele = 0;
                    for(int post_tag_id = 0; post_tag_id < num_postags; ++post_tag_id){
                        ele += beta.get(post_tag_id, i+1) * A.get(tag_id, post_tag_id) * B.get(tag_id, lemme_id);
                    }
                    beta.set(tag_id, i, ele);
                }
            }
            double scale = 0;
            for(int tag_id = 0; tag_id < num_postags; ++tag_id){
                scale += beta.get(tag_id, i);
            }
            scales.set(1, i, scale);
            for(int tag_id = 0; tag_id < num_postags; ++tag_id){
                beta.set(tag_id, i, beta.get(tag_id, i) / scale);
            }
        }

        double res = 0;
        for(int i = 0; i < s.length(); ++i)
            res += Math.log(scales.get(1, i));
        return res;
	}

	/**
	 * Viterbi algorithm for one sentence
	 * v are in log scale, A, B and pi are in the usual scale.
	 */
	private double viterbi(Sentence s) {
        for(int i = 0; i < s.length(); ++i){
            int lemme_id = vocabulary.get(s.getWordAt(i).getLemme()).intValue();
            for(int tag_id = 0; tag_id < num_postags; ++tag_id){
                if(i == 0){
                    v.set(tag_id, i, Math.log(pi.get(0, tag_id) + Math.log(B.get(tag_id, lemme_id))));
                }
                else{
                    double Max = 0;
                    for(int pre_tag_id = 0; pre_tag_id < num_postags; ++ pre_tag_id){
                        double ele = v.get(pre_tag_id, i-1) + Math.log(A.get(pre_tag_id, tag_id)) + Math.log(B.get(tag_id, lemme_id));
                        if(ele > Max){
                            Max = ele;
                            back_pointer.set(tag_id, i, pre_tag_id);
                        }
                    }
                    v.set(tag_id, i, Max);
                }
            }
        }

        double res = 0;
        for(int i = 0; i < num_postags; ++i)
            res = Math.max(res, v.get(i, s.length() - 1));
        return res;
	}

	public static void main(String[] args) throws IOException {
		if (args.length < 3) {
			System.out.println("Expecting at least 3 parameters");
			System.exit(0);
		}
		String labeledFileName = args[0];
		String unlabeledFileName = args[1];
		String predictionFileName = args[2];
		
		String trainingLogFileName = null;
		
		if (args.length > 3) {
			trainingLogFileName = args[3];
		}
		
		// read in labeled corpus
		FileHandler fh = new FileHandler();
		
		ArrayList<Sentence> labeled_corpus = fh.readTaggedSentences(labeledFileName);
		
		ArrayList<Sentence> unlabeled_corpus = fh.readTaggedSentences(unlabeledFileName);

		HMM model = new HMM(labeled_corpus, unlabeled_corpus);

		model.prepareMatrices();
		model.em();
		model.predict();
		model.outputPredictions(predictionFileName);
		
		if (trainingLogFileName != null) {
			model.outputTrainingLog(trainingLogFileName);
		}
	}
}
