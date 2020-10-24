from argparse import ArgumentParser
from os.path import join, split, basename, exists
from os import makedirs, listdir
from glob import glob
from tqdm import tqdm

import torch

from ttd.utils import read_json, write_json, write_txt, read_txt
from ttd.vad_helpers import vad_from_word_level


# This feels very dumb....
def add_builder_specific_args(parser, datasets):
    for ds in datasets:
        if ds.lower() == "maptask":
            from research.datasets.maptask import MaptaskBuilder

            parser = MaptaskBuilder.add_data_specific_args(parser, name=ds)
        elif ds.lower() == "switchboard":
            from research.datasets.switchboard import SwitchboardBuilder

            parser = SwitchboardBuilder.add_data_specific_args(parser, name=ds)
        elif ds.lower() == "persona":
            from research.datasets import PersonaBuilder

            parser = PersonaBuilder.add_data_specific_args(parser, name=ds)
        elif ds.lower() == "dailydialog":
            from research.datasets import DailydialogBuilder

            parser = DailydialogBuilder.add_data_specific_args(parser, name=ds)
        elif ds.lower() == "empathetic":
            from research.datasets import EmpatheticBuilder

            parser = EmpatheticBuilder.add_data_specific_args(parser, name=ds)
        elif ds.lower() == "coached":
            from research.datasets import CoachedBuilder

            parser = CoachedBuilder.add_data_specific_args(parser, name=ds)
        elif ds.lower() == "taskmaster":
            from research.datasets import TaskmasterBuilder

            parser = TaskmasterBuilder.add_data_specific_args(parser, name=ds)
        elif ds.lower() == "multiwoz":
            from research.datasets import MultiwozBuilder

            parser = MultiwozBuilder.add_data_specific_args(parser, name=ds)
        elif ds.lower() == "metalwoz":
            from research.datasets import MetalwozBuilder

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
            from research.datasets import MaptaskBuilder

            tmp_builder = MaptaskBuilder(hparams)
        elif ds.lower() == "switchboard":
            from research.datasets import SwitchboardBuilder

            tmp_builder = SwitchboardBuilder(hparams)
        elif ds.lower() == "persona":
            from research.datasets import PersonaBuilder

            tmp_builder = PersonaBuilder(hparams)
        elif ds.lower() == "dailydialog":
            from research.datasets import DailydialogBuilder

            tmp_builder = DailydialogBuilder(hparams)
        elif ds.lower() == "empathetic":
            from research.datasets import EmpatheticBuilder

            tmp_builder = EmpatheticBuilder(hparams)
        elif ds.lower() == "coached":
            from research.datasets import CoachedBuilder

            tmp_builder = CoachedBuilder(hparams)
        elif ds.lower() == "taskmaster":
            from research.datasets import TaskmasterBuilder

            tmp_builder = TaskmasterBuilder(hparams)
        elif ds.lower() == "multiwoz":
            from research.datasets import MultiwozBuilder

            tmp_builder = MultiwozBuilder(hparams)
        elif ds.lower() == "metalwoz":
            from research.datasets import MetalwozBuilder

            tmp_builder = MetalwozBuilder(hparams)
        else:
            raise NotImplementedError(f"{ds} not implemented")
        builders.append(tmp_builder)
    return builders


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

    def prepare_vad(self):
        if not self.check_if_dir_exists(self.vad_root):
            self._process_vad()

    def prepare_turn_level(self):
        if not self.check_if_dir_exists(self.turn_level_root):
            self._process_turn_level()

    def prepare_word_level(self):
        if not self.check_if_dir_exists(self.word_level_root):
            self._process_word_level()

    def get_audio_path(self, name):
        audio_name = name + self.AUDIO_EXT
        return join(self.audio_root, audio_name)

    def _process_vad(self):
        """
        process vad information which is a list for each channel in the audio with start and end values
        as percentages of the total duration.
        Useful when diffent frame levels are used and so on.
        """
        makedirs(self.vad_root, exist_ok=True)

        # Makes sure that the data we need exists
        self.prepare_word_level()

        # Iterate over the word_level_dialogs and constructs vad base on the duration
        # (using the audio path and the sox to extract the duration of the audio)
        files = glob(join(self.word_level_root, "*.json"))
        for word_level_path in tqdm(files, desc=f"Vad {self.NAME}"):
            json_name = basename(word_level_path)

            word_level_dialog = read_json(word_level_path)
            audio_path = self.get_audio_path(json_name)
            vad = vad_from_word_level(word_level_dialog, audio_path)
            # vad = words_to_vad_percentage(word_level_dialog, audio_path)
            vad_path = join(self.vad_root, json_name)
            torch.save(vad, join(self.vad_root, json_name.replace(".json", ".pt")))

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
    pass
