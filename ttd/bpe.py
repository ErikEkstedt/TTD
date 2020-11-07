from argparse import ArgumentParser
from os.path import join, basename
from os import makedirs
from tqdm import tqdm
from glob import glob
import shutil

import sentencepiece as spm

from ttd.basebuilder import add_builder_specific_args, create_builders
from ttd.utils import write_txt, read_txt, read_json


"""
- [x] Convert data to txt files
- [x] remember all txt files
- [x] define 'sentence-piece' tokens
- [x] train
- [ ] Tokenizer class
    - implement same functions as transformers tokenizer (gpt2) for compatability


Not done and not used in anything yet....

"""

# This is implemented to fit the "basebuilder" style processing
# it would probably be better to just extract text-data from the raw files.
class TextExtraction:
    def __init__(self, hparams):
        super().__init__()
        self.savepath = hparams.savepath
        self.datasets = hparams.datasets
        self.max_sentence_len = hparams.max_sentence_len
        self.only_lower_case = hparams.only_lower_case
        self.builders = create_builders(vars(hparams))

        makedirs(self.savepath, exist_ok=True)
        self.savepath = join(self.savepath, "data")
        makedirs(self.savepath, exist_ok=True)

    def _turn_level_text(self, builder):
        tok_path = builder.prepare_turn_level()
        builder_text = []
        for filepath in tqdm(
            glob(join(builder.turn_level_root, "*.json")),
            desc=f"ToText {builder.NAME}",
        ):
            dialog = read_json(filepath)
            for turn in dialog:
                if self.only_lower_case:
                    turn["text"] = turn["text"].lower()
                builder_text.append(turn["text"])
        return builder_text

    def _word_level_text(self, builder):
        tok_path = builder.prepare_word_level()
        builder_text = []
        for filepath in tqdm(
            glob(join(builder.word_level_root, "*.json")),
            desc=f"ToText {builder.NAME}",
        ):
            dialog = read_json(filepath)
            text = []
            for word in dialog:
                if self.only_lower_case:
                    word["word"] = word["word"].lower()
                text.append(word["word"])
            for i in range(0, len(text), self.max_sentence_len):
                builder_text.append(" ".join(text[i : i + self.max_sentence_len]))
        return builder_text

    def extract(self):
        for i, builder in enumerate(self.builders):
            if builder.NAME.lower() in ["maptask", "switchboard"]:
                builder_text = self._word_level_text(builder)
                # print(builder_text)
                # print(len(builder_text))
                # input()
            else:
                builder_text = self._turn_level_text(builder)
            write_txt(builder_text, join(self.savepath, builder.NAME + ".txt"))


class Tokenizer(object):
    def __init__(
        self,
        model_path=None,
        model_type="bpe",
        vocab_size=10000,
        character_coverage=1.0,
        user_defined_symbols=["<eot>", "<speaker1>", "<speaker2>"],
    ):
        self.model_path = model_path
        self.model_type = model_type
        self.vocab_size = vocab_size
        self.user_defined_symbols = user_defined_symbols
        self.character_coverage = character_coverage

        print(self.model_path)
        print(self.model_type)
        print(self.vocab_size)
        print(self.user_defined_symbols)
        print(self.character_coverage)

        if model_path is not None:
            self.load(model_path)

    def __len__(self):
        return self.sp.vocab_size()

    def load(self, model_path):
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model_path)
        self.vocab_size = self.sp.vocab_size()

    def save(self, savepath):
        new_model_path = join(savepath, basename(self.model_path))
        new_vocab_path = join(savepath, basename(self.vocab_path))
        shutil.move(self.model_path, new_model_path)
        shutil.move(self.vocab_path, new_vocab_path)
        self.model_path = new_model_path
        self.vocab_path = new_model_path
        print("Tokenizer saved! -> ", self.model_path)
        print("Vocab saved! -> ", self.vocab_path)

    def train(self, input):
        spm.SentencePieceTrainer.train(
            input=input,
            model_prefix=self.model_type,
            vocab_size=self.vocab_size,
            user_defined_symbols=self.user_defined_symbols,
            character_coverage=self.character_coverage,
            model_type=self.model_type,
        )
        self.model_path = f"{self.model_type}.model"
        self.vocab_path = f"{self.model_type}.vocab"
        self.load(self.model_path)

    def encode(self, string_or_list):
        """
        string_or_list:     a string or a list of strings to be encoded

        Returns:
            list:           list of indices
        """
        return self.sp.encode(string_or_list)

    def decode(self, idx):
        return self.decode(idx)

    def convert_ids_to_tokens(self, idx):
        if not isinstance(idx, list):
            idx = [idx]

        if isinstance(idx[0], list):
            ret = []
            for tmp_idx in idx:
                ret.append(self.sp.encode_as_pieces(self.sp.decode_ids(tmp_idx)))
        else:
            ret = self.sp.encode_as_pieces(self.sp.decode_ids(idx))
        return ret

    def convert_tokens_to_ids(self, toks):
        return self.sp.encode_as_ids("".join(toks))


def main():
    parser = ArgumentParser()
    parser.add_argument("--max_sentence_len", type=int, default=4000)
    parser.add_argument("--only_lower_case", type=bool, default=True)
    parser.add_argument("--savepath", type=str, default="text_files_for_bpe")

    #  datasets
    parser.add_argument(
        "--datasets",
        nargs="*",
        type=str,
        default=["coached"],
    )
    temp_args, _ = parser.parse_known_args()
    datasets = temp_args.datasets
    parser = add_builder_specific_args(parser, datasets)  # add for all builders

    # sentence piece
    parser.add_argument("--model_type", type=str, default="bpe")
    parser.add_argument("--vocab_size", type=int, default=5000)
    parser.add_argument("--character_coverage", type=float, default=1.0)
    parser.add_argument("--user_define", type=float, default=1.0)
    parser.add_argument(
        "--user_defined_symbols",
        nargs="*",
        type=str,
        default=["<speaker1>", "<speaker2>", "<eot>"],
    )

    args = parser.parse_args()

    # Data -> .txt
    dm = TextExtraction(args)
    dm.extract()
    txt_list = glob(join(dm.savepath, "*.txt"))

    # sentence piece
    tokenizer = Tokenizer(
        model_type=args.model_type,
        vocab_size=args.vocab_size,
        character_coverage=args.character_coverage,
        user_defined_symbols=args.user_defined_symbols,
    )
    tokenizer.train(txt_list)
    tokenizer.save(args.savepath)


def play():
    # txt_list = convert_data_to_txt()
    # spm.SentencePieceTrainer.train( input=txt_list,
    #     model_prefix="bpe",
    #     vocab_size=10000,
    #     character_coverage=1.0,
    #     model_type="bpe",
    # )
    # sp = spm.SentencePieceProcessor()
    # sp.load("bpe.model")
    # print("tokens: ", sp.EncodeAsPieces("This is a test"))
    # print("indices: ", sp.EncodeAsIds("This is a test"))

    # tokenizer = spm.SentencePieceProcessor(model_file='bpe.model')

    tokenizer = Tokenizer(
        model_path="bpe.model",
        model_type="bpe",
        vocab_size=1000,
        character_coverage=1.0,
    )

    print(len(tokenizer))

    tokenizer.train(txt_list)

    text = ["hello there you sexy beast", "you're not so bad yourself"]
    print(text)
    idx = tokenizer.encode(text)
    print(idx)
    toks = tokenizer.convert_ids_to_tokens(idx)
    print(toks)

    t_i = tokenizer.convert_tokens_to_ids(toks)
    print(t_i)

    tokenizer.train(txt_list)
    tokenizer.save("data/tokenizers/")


if __name__ == "__main__":
    main()
