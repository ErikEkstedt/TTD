from os.path import join, exists, basename
from os import makedirs
import subprocess
from argparse import ArgumentParser
from tqdm import tqdm
import re
import csv

from ttd.utils import read_txt, write_json, write_txt, read_json
from ttd.basebuilder import BaseBuilder


def clean_empathetic(t):
    t = re.sub("_comma_", ", ", t)
    t = re.sub("b/c", "because", t)
    t = re.sub("\s([\.\!\?\,])", r"\1", t)  # remove space before punctuation
    t = re.sub("\:\)", "", t)  # remove smiley
    t = re.sub("\:\(", "", t)  # remove smiley
    t = re.sub("\:\/", "", t)  # remove smiley
    t = re.sub("\:\|", "", t)  # remove smiley
    # t = re.sub("[(\:\))|(\:\()]", "", t)  # remove smiley
    t = re.sub("\*", "", t)  # remove smiley
    t = re.sub("\s\-\-\s", " ", t)  # remove "blabla -- blabla"
    t = re.sub("\s\-\s", " ", t)  # remove "blabla - blabla"
    t = re.sub("\s\/\s", " ", t)  # sub kidding/no kidding -> kidding no kidding
    t = re.sub("/", " ", t)
    return t


class EmpatheticBuilder(BaseBuilder):
    URL = "https://dl.fbaipublicfiles.com/parlai/empatheticdialogues/empatheticdialogues.tar.gz"
    NAME = "Empathetic"

    def _process_word_level(self):
        raise NotImplementedError("{self.NAME} does not contain word-level timings")

    def download_audio(self):
        raise NotImplementedError("Taskmaster has no audio")

    def download_text(self):
        if not exists(join(self.raw_data_root, "train.csv")):
            name = basename(self.URL)
            zip_name = join(self.root, name)
            if not exists(zip_name):
                # downloads data/empathetic/personachat.tgz
                print("Download data")
                # wget(self.URL, zip_name)
                subprocess.call(["wget", "-q", "-O", zip_name, self.URL])
            print("Extract data")
            zip_file = join(self.root, name)
            subprocess.call(["tar", "-zxvf", zip_file])
            subprocess.call(["mv", "empatheticdialogues", self.raw_data_root])
            print(f"Annotations -> {self.raw_data_root}")

    def _process_turn_level(self):
        makedirs(self.turn_level_root, exist_ok=True)

        self.download_text()

        train_filepaths = []
        val_filepaths = []
        test_filepaths = []
        # header index
        # 0:   conv_id
        # 1:   utterance_idx
        # 2:   context
        # 3:   prompt
        # 4:   speaker_idx
        # 5:   utterance
        # 6:   selfeval
        # 7:   tags
        omitted = 0
        n = 0
        files = ["train.csv", "valid.csv", "test.csv"]
        for filename in files:
            data = open(join(self.raw_data_root, filename)).readlines()
            filepaths = []
            dialog = []
            omit_next_dialog = False
            last_conv_id = data[1].strip().split(",")[0]
            for i in tqdm(
                range(1, len(data)), desc=f"{self.NAME} Turn-level ({filename})"
            ):  # skip header
                row = data[i].strip().split(",")
                conv_id = row[0]
                utt_idx = int(row[1])
                speaker_id = (
                    utt_idx + 1
                ) % 2  # starts on utt_idx = 1 -> speaker_id = 0
                utt = {
                    "text": clean_empathetic(row[5]),
                    "speaker_id": speaker_id,
                    "start": utt_idx,
                    "emotion": row[2],
                    "id": conv_id,
                }
                if "|" in utt["text"]:
                    omit_next_dialog = True
                if last_conv_id == conv_id:
                    dialog.append(utt)
                else:
                    if not omit_next_dialog:
                        savename = f"emp{n}.json"
                        write_json(dialog, join(self.turn_level_root, savename))
                        filepaths.append(savename)
                        n += 1
                    else:
                        omitted += 1
                    omit_next_dialog = False
                    dialog = [utt]
                    last_conv_id = conv_id
            if "train" in filename:
                write_txt(filepaths, join(self.root, "train.txt"))
            elif "valid" in filename:
                write_txt(filepaths, join(self.root, "val.txt"))
            else:
                write_txt(filepaths, join(self.root, "test.txt"))
        print(f"Omitted: {omitted} dialogs")


if __name__ == "__main__":

    parser = ArgumentParser()
    parser = EmpatheticBuilder.add_data_specific_args(parser, name="empathetic")
    args = parser.parse_args()
    hparams = vars(args)
    builder = EmpatheticBuilder(hparams)
    builder.prepare_turn_level()

    file = join(builder.turn_level_root, builder.val_filepaths[0])
    f = read_json(file)
    print(f)
