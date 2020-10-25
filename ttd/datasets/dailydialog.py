from argparse import ArgumentParser
from os.path import join, exists, basename
from os import makedirs
import subprocess
from tqdm import tqdm
import re

from ttd.utils import read_txt, write_json, write_txt, wget, read_json
from ttd.basebuilder import BaseBuilder


def clean_daily(text):
    text = re.sub("\s’\s", "'", text)
    text = re.sub("\s([?!'.,])", r"\1", text).strip()
    return text


class DailydialogBuilder(BaseBuilder):
    URL = "http://yanran.li/files/ijcnlp_dailydialog.zip"
    NAME = "DailyDialog"
    EOT = "__eou__"

    def _process_word_level(self):
        raise NotImplementedError("{self.NAME} does not contain word-level timings")

    def download_audio(self):
        raise NotImplementedError("Taskmaster has no audio")

    def download_text(self):
        train_data_path = join(self.raw_data_root, "train/dialogues_train.txt")
        if not exists(train_data_path):
            name = basename(self.URL)
            zip_name = join(self.root, name)
            if not exists(zip_name):
                print("Download data")
                wget(self.URL, zip_name)

            print("Extract data")
            subprocess.call(["unzip", "-d", self.root, zip_name])
            unzipped_name = join(self.root, name.replace(".zip", ""))
            subprocess.call(["mv", unzipped_name, self.raw_data_root])
            # unzip train, validation, test
            subprocess.call(
                [
                    "unzip",
                    "-d",
                    self.raw_data_root,
                    join(self.raw_data_root, "train.zip"),
                ]
            )
            subprocess.call(
                [
                    "unzip",
                    "-d",
                    self.raw_data_root,
                    join(self.raw_data_root, "validation.zip"),
                ]
            )
            subprocess.call(
                [
                    "unzip",
                    "-d",
                    self.raw_data_root,
                    join(self.raw_data_root, "test.zip"),
                ]
            )

    def _process_turn_level(self):
        """
        Iterate over rows in txt file and clean the text (e.g: don ´ t => don't, tonight ? => tonight?), add speaker_id,
        dialog_act (integer) and emotion (integer).

        save each dialog in a json file:
            [
            {'text', 'speaker_id', 'start', 'act', 'emotion'},
            ...,
            {'text', 'speaker_id', 'start', 'act', 'emotion'},
            ]
        """

        makedirs(self.turn_level_root, exist_ok=True)
        self.download_text()  # make sure data is accessable

        dialog_num = 0
        print(f"{self.NAME} Turn-level")
        for split in ["train", "test", "validation"]:
            dialog_text = read_txt(
                join(self.raw_data_root, split, f"dialogues_{split}.txt")
            )
            dialog_emotion = read_txt(
                join(self.raw_data_root, split, f"dialogues_emotion_{split}.txt")
            )
            dialog_act = read_txt(
                join(self.raw_data_root, split, f"dialogues_act_{split}.txt")
            )

            filepaths = []
            for text, emotion, act in tqdm(
                zip(dialog_text, dialog_emotion, dialog_act),
                desc=split,
                total=len(dialog_act),
            ):
                text = text.strip().split(self.EOT)[:-1]
                emotion = emotion.split()
                act = act.split()
                conversation = []
                for i, (t, e, a) in enumerate(zip(text, emotion, act)):
                    if i % 2 == 0:
                        speaker_id = 0
                    else:
                        speaker_id = 1
                    conversation.append(
                        {
                            "text": clean_daily(t),
                            "speaker_id": speaker_id,
                            "act": int(a),
                            "emotion": int(e),
                            "start": i,
                        }
                    )
                savename = f"dd{dialog_num}.json"
                write_json(conversation, join(self.turn_level_root, savename))
                filepaths.append(savename)
                dialog_num += 1

            if split == "validation":
                write_txt(filepaths, join(self.root, f"val.txt"))
            else:
                write_txt(filepaths, join(self.root, f"{split}.txt"))


if __name__ == "__main__":

    parser = ArgumentParser()
    parser = DailydialogBuilder.add_data_specific_args(parser, name="dailydialog")
    args = parser.parse_args()
    hparams = vars(args)
    builder = DailydialogBuilder(hparams)
    builder.prepare_turn_level()

    file = join(builder.turn_level_root, builder.val_filepaths[0])
    print(read_json(file))
