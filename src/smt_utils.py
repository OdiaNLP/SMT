import nltk
from nltk.translate import AlignedSent
from nltk.translate.ibm2 import (
    IBMModel2,
    Model2Counts
)
from tqdm import tqdm


class IBMModel2WithProgressbar(IBMModel2):
    def __init__(
            self,
            sentence_aligned_corpus,
            iterations,
            probability_tables=None
    ):
        """
        IBM Model 2 with progress bar for training
        """
        super(IBMModel2WithProgressbar, self).__init__(
            sentence_aligned_corpus,
            iterations, probability_tables
        )

    def train(self, parallel_corpus):
        counts = Model2Counts()
        for aligned_sentence in tqdm(parallel_corpus, unit=' samples'):
            src_sentence = [None] + aligned_sentence.mots
            trg_sentence = ['UNUSED'] + aligned_sentence.words  # 1-indexed
            l = len(aligned_sentence.mots)
            m = len(aligned_sentence.words)

            # E step (a): Compute normalization factors to weigh counts
            total_count = self.prob_all_alignments(src_sentence, trg_sentence)

            # E step (b): Collect counts
            for j in range(1, m + 1):
                t = trg_sentence[j]
                for i in range(0, l + 1):
                    s = src_sentence[i]
                    count = self.prob_alignment_point(i, j, src_sentence, trg_sentence)
                    normalized_count = count / total_count[t]

                    counts.update_lexical_translation(normalized_count, s, t)
                    counts.update_alignment(normalized_count, i, j, l, m)

        # M step: Update probabilities with maximum likelihood estimates
        self.maximize_lexical_translation_probabilities(counts)
        self.maximize_alignment_probabilities(counts)


def train_ibmmodel2(src_text, trg_text, iterations=5):
    """
    train IBM model 2
    :param src_text: (list) src text
    :param trg_text: (list) trg text
    :param iterations: (int) number of iterations to run training algorithm
    :return: trained IBM model
    """
    if len(src_text) != len(trg_text):
        raise AssertionError("different numbers of samples in src and trg")
    aligned_text = []
    for src_sample, trg_sample in zip(src_text, trg_text):
        al_sent = AlignedSent(src_sample, trg_sample)
        aligned_text.append(al_sent)
    ibm_model = IBMModel2WithProgressbar(aligned_text, iterations)
    return ibm_model


def translate(ibm_model, src_tokens):
    translation_tokens = []
    for tok in src_tokens:
        probs = ibm_model.translation_table[tok]
        if len(probs) == 0:
            continue
        sorted_words = sorted(
            [(k, v) for k, v in probs.items()],
            key=lambda x: x[1],
            reverse=True
        )
        top_token = sorted_words[1][0]
        if top_token is not None:
            translation_tokens.append(top_token)
    return translation_tokens


def tokenize_en(sent, lowercase=False):
    toks = nltk.word_tokenize(sent)
    return [tok.lower() for tok in toks] if lowercase else toks


def tokenize_od(sent):
    return sent.split()


def detokenize_od(toks):
    return ' '.join(toks)
