from argparse import ArgumentParser
from os.path import join, split, basename, exists
from os import makedirs, listdir
from glob import glob
from tqdm import tqdm
import time
import shutil

import torch

from ttd.utils import read_json, write_json, write_txt, read_txt
from ttd.vad_helpers import vad_from_word_level
from ttd.POS import extract_turn_level_pos
from ttd.tokenizer_helpers import (
    tokenize_word_level_dialog,
    tokenize_turn_level_dialog,
    add_explicit_turn_shift_token,
    chunk_tokenized_dialog,
    tokenizer_info,
)


# This feels very dumb....
def add_builder_specific_args(parser, datasets):
    for ds in datasets:
        if ds.lower() == "maptask":
            from ttd.datasets.maptask import MaptaskBuilder

            parser = MaptaskBuilder.add_data_specific_args(parser, name=ds)
        elif ds.lower() == "switchboard":
            from ttd.datasets.switchboard import SwitchboardBuilder

            parser = SwitchboardBuilder.add_data_specific_args(parser, name=ds)
        elif ds.lower() == "persona":
            from ttd.datasets import PersonaBuilder

            parser = PersonaBuilder.add_data_specific_args(parser, name=ds)
        elif ds.lower() == "dailydialog":
            from ttd.datasets import DailydialogBuilder

            parser = DailydialogBuilder.add_data_specific_args(parser, name=ds)
        elif ds.lower() == "empathetic":
            from ttd.datasets import EmpatheticBuilder

            parser = EmpatheticBuilder.add_data_specific_args(parser, name=ds)
        elif ds.lower() == "coached":
            from ttd.datasets import CoachedBuilder

            parser = CoachedBuilder.add_data_specific_args(parser, name=ds)
        elif ds.lower() == "taskmaster":
            from ttd.datasets import TaskmasterBuilder

            parser = TaskmasterBuilder.add_data_specific_args(parser, name=ds)
        elif ds.lower() == "multiwoz":
            from ttd.datasets import MultiwozBuilder

            parser = MultiwozBuilder.add_data_specific_args(parser, name=ds)
        elif ds.lower() == "metalwoz":
            from ttd.datasets import MetalwozBuilder

            parser = MetalwozBuilder.add_data_specific_args(parser, name=ds)
        else:
            raise NotImplementedError(f"{ds} not implemented")
    return parser


# This feels very dumb....
def create_builders(hparams):
    """
    Used in DataModule which leverages several different datasets.
    """
    builders = []
    for ds in hparams["datasets"]:
        if ds.lower() == "maptask":
            from ttd.datasets import MaptaskBuilder

            tmp_builder = MaptaskBuilder(hparams)
        elif ds.lower() == "switchboard":
            from ttd.datasets import SwitchboardBuilder

            tmp_builder = SwitchboardBuilder(hparams)
        elif ds.lower() == "persona":
            from ttd.datasets import PersonaBuilder

            tmp_builder = PersonaBuilder(hparams)
        elif ds.lower() == "dailydialog":
            from ttd.datasets import DailydialogBuilder

            tmp_builder = DailydialogBuilder(hparams)
        elif ds.lower() == "empathetic":
            from ttd.datasets import EmpatheticBuilder

            tmp_builder = EmpatheticBuilder(hparams)
        elif ds.lower() == "coached":
            from ttd.datasets import CoachedBuilder

            tmp_builder = CoachedBuilder(hparams)
        elif ds.lower() == "taskmaster":
            from ttd.datasets import TaskmasterBuilder

            tmp_builder = TaskmasterBuilder(hparams)
        elif ds.lower() == "multiwoz":
            from ttd.datasets import MultiwozBuilder

            tmp_builder = MultiwozBuilder(hparams)
        elif ds.lower() == "metalwoz":
            from ttd.datasets import MetalwozBuilder

            tmp_builder = MetalwozBuilder(hparams)
        else:
            raise NotImplementedError(f"{ds} not implemented")
        builders.append(tmp_builder)
    return builders


# Superclass used for all datasets
class BaseBuilder(object):
    def __init__(self, hparams):
        if not isinstance(hparams, dict):
            hparams = vars(hparams)

        self.hparams = hparams
        self.set_paths()
        self.train_filepaths = self.get_filepaths(split="train")
        self.val_filepaths = self.get_filepaths(split="val")
        self.test_filepaths = self.get_filepaths(split="test")

    def set_paths(self):
        self.root = self.hparams[f"root_{self.NAME.lower()}"]
        self.raw_data_root = join(self.root, "raw_data")
        self.turn_level_root = join(self.root, "dialogs_turn_level")
        self.word_level_root = join(self.root, "dialogs_word_level")
        self.tokenized_turn_level_root = join(self.root, "tokenized_turn_level")
        self.tokenized_word_level_root = join(self.root, "tokenized_word_level")
        self.audio_root = join(self.root, "audio")
        self.vad_root = join(self.root, "VAD")
        self.pos_root = join(self.root, "POS")
        makedirs(self.root, exist_ok=True)

    def get_filepaths(self, split="train"):
        filepaths = self.hparams.get(f"{split}_split_{self.NAME.lower()}", None)
        if isinstance(filepaths, str):
            if exists(filepaths):
                filepaths = read_txt(filepaths)
        return filepaths

    def check_if_dir_exists(self, dir_path, file_ext=None):
        if not exists(dir_path):
            return False

        if file_ext is None:
            files = listdir(dir_path)
        else:
            files = glob(join(dir_path, "*" + file_ext))
        if len(files) == 0:
            return False
        return True

    def prepare_pos(self):
        if not self.check_if_dir_exists(self.pos_root):
            makedirs(self.pos_root, exist_ok=True)

            # Makes sure that the data we need exists
            self.prepare_turn_level()

            # Iterate over the turn_level_dialogs and constructs vad base on the duration
            # (using the audio path and the sox to extract the duration of the audio)
            files = glob(join(self.turn_level_root, "*.json"))
            for turn_level_path in tqdm(files, desc=f"{self.NAME} POS"):
                turn_level_dialog = read_json(turn_level_path)
                pos, words = extract_turn_level_pos(turn_level_dialog)
                write_json(
                    {"pos": pos, "words": words},
                    join(self.pos_root, basename(turn_level_path)),
                )
        return self.pos_root

    def prepare_vad(self):
        """
        process vad information which is a list for each channel in the audio with start and end values
        as percentages of the total duration.
        Useful when diffent frame levels are used and so on.
        """
        if not self.check_if_dir_exists(self.vad_root):
            makedirs(self.vad_root, exist_ok=True)

            # Makes sure that the data we need exists
            self.prepare_word_level()

            # Iterate over the word_level_dialogs and constructs vad base on the duration
            # (using the audio path and the sox to extract the duration of the audio)
            files = glob(join(self.word_level_root, "*.json"))
            for word_level_path in tqdm(files, desc=f"{self.NAME} VAD"):
                json_name = basename(word_level_path)

                word_level_dialog = read_json(word_level_path)
                audio_path = self.get_audio_path(json_name)
                vad = vad_from_word_level(word_level_dialog, audio_path)
                # vad = words_to_vad_percentage(word_level_dialog, audio_path)
                vad_path = join(self.vad_root, json_name)
                torch.save(vad, join(self.vad_root, json_name.replace(".json", ".pt")))
            return self.vad_root

    def get_all_filepaths(self):
        return self.train_filepaths + self.val_filepaths + self.test_filepaths

    def prepare_turn_level(self):
        if not self.check_if_dir_exists(self.turn_level_root):
            self._process_turn_level()
        return self.turn_level_root

    def prepare_word_level(self):
        if not self.check_if_dir_exists(self.word_level_root):
            self._process_word_level()
        return self.word_level_root

    def get_audio_path(self, name):
        audio_name = name + self.AUDIO_EXT
        return join(self.audio_root, audio_name)

    def prepare_turn_level_tokens(self, tokenizer):
        if not self.check_if_dir_exists(self.tokenized_turn_level_root, ".json"):
            self.prepare_turn_level()

            makedirs(self.tokenized_turn_level_root, exist_ok=True)

            # TOKENIZER SANITY CHECK
            _ = tokenizer_info(
                tokenizer, self.tokenized_turn_level_root
            )  # Save tokenizer info for checks

            t = time.time()
            broken = 0
            broken_files = []
            for turn_level_path in tqdm(
                glob(join(self.turn_level_root, "*.json")),
                desc=f"Tokenizing Turn-level {self.NAME}",
            ):
                turn_level_dialog = read_json(turn_level_path)

                (
                    input_ids,
                    speaker_ids,
                    word_ids,
                    starts,
                    ends,
                ) = tokenize_turn_level_dialog(
                    turn_level_dialog, tokenizer, remove_punctuation=True
                )

                data = {
                    "input_ids": input_ids,
                    "speaker_ids": speaker_ids,
                    "word_ids": word_ids,
                }

                if len(starts) > 0:
                    data["starts"] = starts

                if len(ends) > 0:
                    data["ends"] = ends

                write_json(
                    data,
                    join(self.tokenized_turn_level_root, basename(turn_level_path)),
                )

            t = time.time() - t
            print(f"{self.NAME} tokenization took {round(t, 1)} seconds")
            print(f"{self.NAME} broken", broken)
            write_txt(broken_files, join(self.root, "broken_tokenize.txt"))
        return self.tokenized_turn_level_root

    def prepare_word_level_tokens(self, tokenizer):
        if not self.check_if_dir_exists(self.tokenized_word_level_root):
            self.prepare_word_level()

            makedirs(self.tokenized_word_level_root, exist_ok=True)

            # TOKENIZER SANITY CHECK
            _ = tokenizer_info(
                tokenizer, self.tokenized_word_level_root
            )  # Save tokenizer info for checks

            desc = f"Tokenizing Word-level {self.NAME}"
            t = time.time()
            broken = 0
            broken_files = []
            for word_level_path in tqdm(
                glob(join(self.word_level_root, "*.json")), desc=desc
            ):
                json_name = basename(word_level_path)
                word_level_dialog = read_json(word_level_path)

                (
                    input_ids,
                    speaker_ids,
                    word_ids,
                    starts,
                    ends,
                ) = tokenize_word_level_dialog(
                    word_level_dialog,
                    tokenizer,
                )
                write_json(
                    {
                        "input_ids": input_ids,
                        "speaker_ids": speaker_ids,
                        "starts": starts,
                        "ends": ends,
                        "word_ids": word_ids,
                    },
                    join(self.tokenized_word_level_root, json_name),
                )

            t = time.time() - t
            print(f"{self.NAME} tokenization took {round(t, 1)} seconds")
            print(f"{self.NAME} broken", broken)
            write_txt(broken_files, join(self.root, "broken_tokenize.txt"))
        return self.tokenized_word_level_root

    def prepare_explicit_turn_level_tokens(self, tokenizer, EOT_token_id=None):
        """
        loads all tokenized turn-level dialogs and inserts, either a special EOT_token_id (if not None)
        or the index of the next speaker token, in between the turns.
        """

        tokenized_explicit_turn_path = self.get_tokenized_root(
            level="turn", explicit_turns=True, EOT_token_id=EOT_token_id, chunk_size=-1
        )

        if not self.check_if_dir_exists(tokenized_explicit_turn_path, ".json"):
            self.prepare_turn_level_tokens(tokenizer)  # check the necessary data exists

            makedirs(tokenized_explicit_turn_path, exist_ok=True)

            # Copy tokenizer info
            src = join(self.tokenized_turn_level_root, "tokenizer_info")
            dst = join(tokenized_explicit_turn_path, "tokenizer_info")
            shutil.copy(src, dst)

            tok_files = glob(join(self.tokenized_turn_level_root, "*.json"))
            for tokenized_turn_level_path in tqdm(
                tok_files, desc=f"{self.NAME} Explicit turns"
            ):
                tokenized_turn_level_dialog = read_json(tokenized_turn_level_path)
                explicit_turns = add_explicit_turn_shift_token(
                    tokenized_turn_level_dialog, EOT_token_id
                )
                json_name = basename(tokenized_turn_level_path)
                write_json(
                    explicit_turns, join(tokenized_explicit_turn_path, json_name)
                )
        return tokenized_explicit_turn_path

    def prepare_explicit_word_level_tokens(self, tokenizer, EOT_token_id=None):
        """
        loads all tokenized turn-level dialogs and inserts, either a special EOT_token_id (if not None)
        or the index of the next speaker token, in between the turns.
        """

        tokenized_explicit_word_path = self.get_tokenized_root(
            level="word", explicit_turns=True, EOT_token_id=EOT_token_id, chunk_size=-1
        )

        if not self.check_if_dir_exists(tokenized_explicit_word_path, ".json"):
            self.prepare_word_level_tokens(tokenizer)  # check the necessary data exists

            makedirs(tokenized_explicit_word_path, exist_ok=True)

            # Copy tokenizer info
            src = join(self.tokenized_word_level_root, "tokenizer_info")
            dst = join(tokenized_explicit_word_path, "tokenizer_info")
            shutil.copy(src, dst)

            tok_files = glob(join(self.tokenized_word_level_root, "*.json"))
            for tokenized_turn_level_path in tqdm(
                tok_files, desc=f"{self.NAME} Explicit turns"
            ):
                tokenized_turn_level_dialog = read_json(tokenized_turn_level_path)
                explicit_turns = add_explicit_turn_shift_token(
                    tokenized_turn_level_dialog, EOT_token_id
                )
                json_name = basename(tokenized_turn_level_path)
                write_json(
                    explicit_turns, join(tokenized_explicit_word_path, json_name)
                )
        return tokenized_explicit_word_path

    def prepare_chunked_tokens(
        self, tokenized_path, chunk_size, overlap, keep_length, sep="_#"
    ):
        assert chunk_size > 0, "chunk size must be larger than 0"
        tokenized_chunk_path = tokenized_path + f"_chunk-{chunk_size}"

        if not self.check_if_dir_exists(tokenized_chunk_path, ".json"):
            print(f"Chunk {self.NAME} -> {chunk_size}")
            makedirs(tokenized_chunk_path, exist_ok=True)

            # Copy tokenizer used
            # tokenizer_info(tokenizer, self.tokenized_turn_level_root)
            src = join(tokenized_path, "tokenizer_info")
            dst = join(tokenized_chunk_path, "tokenizer_info")
            shutil.copy(src, dst)

            tokenized_files = glob(join(tokenized_path, "*.json"))
            for json_path in tqdm(tokenized_files, desc=f"{self.NAME} Chunk"):
                tokenized_dialog = read_json(json_path)
                chunked_dialogs = chunk_tokenized_dialog(
                    tokenized_dialog, chunk_size, overlap, keep_length
                )

                # Save the chunked files
                name = basename(json_path).replace(".json", "")
                for i, chunked_dialog in enumerate(chunked_dialogs):
                    tmp_name = name
                    if i > 0:
                        tmp_name += sep + str(i)
                    write_json(
                        chunked_dialog, join(tokenized_chunk_path, tmp_name + ".json")
                    )

        print("Chunk size: ", chunk_size)
        return tokenized_chunk_path

    def transform_split_filepaths_with_chunks(self, chunked_path, sep="_#"):
        chunked_files = glob(join(chunked_path, "*.json"))

        train_extended = []
        val_extended = []
        test_extended = []
        for f in chunked_files:
            path = split(f)[0]
            filename = basename(f)
            name = filename.replace(".json", "").split(sep)[0]
            orig_name = name + ".json"
            if orig_name in self.train_filepaths:
                train_extended.append(filename)
            if orig_name in self.test_filepaths:
                val_extended.append(filename)
            if orig_name in self.val_filepaths:
                test_extended.append(filename)

        print(self.NAME)
        print(f"Train {len(self.train_filepaths)} -> {len(train_extended)}")
        print(f"val {len(self.val_filepaths)} -> {len(val_extended)}")
        print(f"test {len(self.test_filepaths)} -> {len(test_extended)}")
        print("-" * 50)
        self.train_filepaths = train_extended
        self.val_filepaths = val_extended
        self.test_filepaths = test_extended

    def get_tokenized_root(
        self, level="turn", explicit_turns=False, EOT_token_id=None, chunk_size=-1
    ):
        if level == "turn":
            path = self.tokenized_turn_level_root
        else:
            path = self.tokenized_word_level_root

        if explicit_turns:
            path += "_explicit_turns"

        if EOT_token_id is not None:
            path += f"_ts-{EOT_token_id}"

        if chunk_size > 0:
            path += f"_chunk-{chunk_size}"
        return path

    @staticmethod
    def add_data_specific_args(parent_parser, name="name"):
        """ Specify the hyperparams for this LightningModule """
        parser = ArgumentParser(
            parents=[parent_parser], conflict_handler="resolve", add_help=False
        )
        parser.add_argument(f"--root_{name}", type=str, default=f"data/{name}")
        parser.add_argument(
            f"--train_split_{name}", type=str, default=f"data/{name}/train.txt"
        )
        parser.add_argument(
            f"--val_split_{name}", type=str, default=f"data/{name}/val.txt"
        )
        parser.add_argument(
            f"--test_split_{name}", type=str, default=f"data/{name}/test.txt"
        )
        return parser


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument(
        "--datasets",
        nargs="*",
        type=str,
        default=["maptask"],
    )
    datasets = parser.parse_args().datasets
    parser = add_builder_specific_args(parser, datasets)  # add for all builders
    args = parser.parse_args()

    for k, v in vars(args).items():
        print(f"{k}: {v}")

    builders = create_builders(vars(args))
    tokenizer = torch.load(
        "turngpt_mini/turngpt/runs/TurnGPTpretrained/pretrained/deepvoice/version_0/tokenizer.pt"
    )
