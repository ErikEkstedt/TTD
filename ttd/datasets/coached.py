from argparse import ArgumentParser
from os.path import join, exists, basename
from os import makedirs
import random

from ttd.basebuilder import BaseBuilder
from ttd.dialog_helpers import join_consecutive_utterances
from ttd.utils import (
    read_json,
    write_json,
    write_txt,
    wget,
    get_duration_sox,
    get_sample_rate_sox,
)


"""
Verbal DataModule calls:

    dm.prepare_data(audio=False, vad=False)
    dm._prepare_tokens(
        tokenizer,
        max_sequence=0,
        sequence_overlap=0,
        min_sequence_keep=float("inf"),
        explicit_turns=False,
        )
"""

random.seed(10)  # for split creation


class CoachedBuilder(BaseBuilder):
    URL = "https://storage.googleapis.com/dialog-data-corpus/CCPE-M-2019/data.json"
    README = "https://storage.googleapis.com/dialog-data-corpus/CCPE-M-2019/README.txt"
    NAME = "Coached"
    SPLIT = {"train": 0.9, "test": 0.05}  # validation is remaining

    def _process_word_level(self):
        raise NotImplementedError("{self.NAME} does not contain word-level timings")

    def download_audio(self):
        raise NotImplementedError("Taskmaster has no audio")

    def download_text(self):
        """
        downloads data.json -> self.annotation_root/data.json
        """
        makedirs(self.raw_data_root, exist_ok=True)
        if not exists(join(self.raw_data_root, "data.json")):
            zip_name = join(self.raw_data_root, basename(self.URL))
            if not exists(zip_name):
                print("Download", self.NAME)
                wget(self.URL, zip_name)

    def _process_turn_level(self):
        """
        There is no pre-defined splits so we set random seed (see imports) and split all domains into train/val/test
        """
        makedirs(self.turn_level_root, exist_ok=True)

        self.download_text()  # make sure the data is accesable

        total, skipped = 0, 0
        filenames = []
        data = read_json(join(self.raw_data_root, "data.json"))
        for dialog in data:
            filename = dialog["conversationId"] + ".json"
            conversation = []
            for utt in dialog["utterances"]:
                speaker_id = 1
                if utt["speaker"] == "ASSISTANT":
                    speaker_id = 0
                conversation.append(
                    {
                        "text": utt["text"],
                        "speaker_id": speaker_id,
                        "start": utt["index"],
                    }
                )
            conversation = join_consecutive_utterances(conversation)
            if len(conversation) > 1:
                write_json(conversation, join(self.turn_level_root, filename))
                filenames.append(filename)
                total += 1
            else:
                skipped += 1

        train_filepaths, val_filepaths, test_filepaths = self._create_splits(filenames)
        print(self.NAME)
        print("Skipped: ", skipped)
        print("Total dialogs: ", total)
        print("Train: ", len(train_filepaths))
        print("Val: ", len(val_filepaths))
        print("Test: ", len(test_filepaths))
        write_txt(train_filepaths, join(self.root, "train.txt"))
        write_txt(val_filepaths, join(self.root, "val.txt"))
        write_txt(test_filepaths, join(self.root, "test.txt"))

    def _create_splits(self, filenames):
        random.shuffle(filenames)
        n_train = int(len(filenames) * self.SPLIT["train"])
        n_test = int(len(filenames) * self.SPLIT["test"])
        n_val = len(filenames) - n_train - n_test
        train = filenames[:n_train]
        test = filenames[n_train : n_train + n_test]
        val = filenames[n_train + n_test :]
        return train, val, test


if __name__ == "__main__":

    parser = ArgumentParser()
    parser = CoachedBuilder.add_data_specific_args(parser, name="coached")
    args = parser.parse_args()
    hparams = vars(args)
    builder = CoachedBuilder(hparams)
    builder.prepare_turn_level()

    file = join(builder.turn_level_root, builder.val_filepaths[0])
    print(read_json(file))
