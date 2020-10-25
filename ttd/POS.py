from collections import defaultdict
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize

from ttd.tokenizer_helpers import remove_punctuation_capitalization


SPEAKER_TOKEN = "SPEAKER"
INTERJECTIONS = ["uhuh", "uh", "oh", "eh", "um", "hm", "ehm"]
PUNCTUATION = ".,:?!;()[]$"


def turn_level_pos(text):
    clean_p, clean_w = [], []
    for w, p in pos_tag(word_tokenize(text)):
        cont = False
        for _w in w:
            if _w in PUNCTUATION:
                cont = True
                break
        if cont:
            continue

        # if w in PUNCTUATION:
        #     continue
        if w in INTERJECTIONS:
            p = "UH"
        if w != "":  # omit empty
            clean_w.append(w)
            clean_p.append(p)
    clean_p.append(SPEAKER_TOKEN)  # last turn
    clean_w.append(SPEAKER_TOKEN)  # last turn
    return clean_p, clean_w


def extract_turn_level_pos(dialog, remove_punctuation=True):
    all_pos, all_words = [], []
    for turn in dialog:
        if remove_punctuation:
            turn["text"] = remove_punctuation_capitalization(turn["text"])
        p, w = turn_level_pos(turn["text"])
        all_pos += p
        all_words += w
    return all_pos, all_words


class GramTrainer(object):
    def __init__(self):
        # The mapping from words to identifiers.
        self.index = {}

        # The mapping from identifiers to words.
        self.word = {}

        # An array holding the unigram counts.
        self.unigram_count = defaultdict(int)

        """
        The bigram counts. Since most of these are zero (why?), we store these
        in a hashmap rather than an array to save space (and since it is impossible
        to create such a big array anyway).
        """
        self.bigram_count = defaultdict(lambda: defaultdict(int))
        self.trigram_count = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

        # Number of unique words (word forms) in the training corpus.
        self.unique_words = 0
        # The total number of words in the training corpus.
        self.total_words = 0

    def process(self, tokens):
        # The identifier of the previous word processed.
        self.last_index = -1  # reset between dialogs
        self.second_last_index = -1  # reset between dialogs
        for token in tokens:
            self.process_token(token)

    def process_token(self, token):
        """
        Processes one word in the training corpus, and adjusts the unigram and
        bigram counts.
        :param token: The current word to be processed.
        """
        self.total_words += 1

        # Add new indices if token has not been seen before
        ind = self.index.get(token, None)
        if ind == None:
            ind = self.unique_words
            self.word[self.unique_words] = token
            self.index[token] = self.unique_words
            self.unique_words += 1

        # Add to unigram count
        self.unigram_count[ind] += 1

        # Add to bigram count
        if self.last_index >= 0:
            self.bigram_count[self.last_index][ind] += 1

        # Add to trigram count
        if self.second_last_index >= 0:
            self.trigram_count[self.second_last_index][self.last_index][ind] += 1

        # Update
        self.second_last_index = self.last_index
        self.last_index = ind

    def finalize(self):
        sp_ind = self.index["SPEAKER"]

        self.unigram_prob = {}
        for w_ind, n in self.unigram_count.items():
            w = self.word[w_ind]
            self.unigram_prob[w] = n / self.total_words

        # Bigram
        # bigram_speaker_prob % = [w1, Speaker]
        self.bigram_speaker_prob = {}
        for w1_ind, w2_count in self.bigram_count.items():
            w1 = self.word[w1_ind]
            tot = 0
            for w2, n in w2_count.items():
                tot += n
            sp_count = w2_count[sp_ind]
            self.bigram_speaker_prob[w1] = sp_count / tot

        # Trigram
        # trigram_speaker_prob % = [w1, w2, Speaker]
        self.trigram_speaker_prob = {}
        for w1_ind, w2_count in self.trigram_count.items():
            w1 = self.word[w1_ind]
            self.trigram_speaker_prob[w1] = {}
            for w2_ind, w3_count in w2_count.items():
                w2 = self.word[w2_ind]
                bi_tot = 0
                for w3_ind, n in w3_count.items():
                    bi_tot += n
                sp_count = w3_count[sp_ind]
                self.trigram_speaker_prob[w1][w2] = sp_count / bi_tot
        return {
            "uni": self.unigram_prob,
            "bi": self.bigram_speaker_prob,
            "tri": self.trigram_speaker_prob,
        }

    def model(self):
        _ = self.finalize()
        return {
            "index": self.index,
            "word": self.word,
            "unigram_count": self.unigram_count,
            "unique_words": self.unique_words,
            "total_words": self.total_words,
            "unigram_probs": self.unigram_prob,
            "bigram_probs": self.bigram_prob,
            "trigram_probs": self.trigram_prob,
        }


def debug():
    from research.utils import read_json

    texts = [
        "hello there ((lady))?",
        "hello to you sir )",
        "how is your family doing these days?" " ",
    ]
    for text in texts:
        p, w = turn_level_pos(text)
        for pp, ww in zip(p, w):
            print(ww, pp)
        print("-" * 50)

    # text = "hello, my name is erik and I feel like you should know what you're doing"
    # p, w = _pos(text)
    # for pp, ww in zip(p, w):
    #     print(ww, pp)

    dialog = read_json("data/persona/dialogs/persona0.json")

    pos, words = extract_turn_level_pos(dialog)
    for ww, pp in zip(words, pos):
        print(ww, "->", pp)


if __name__ == "__main__":
    debug()
