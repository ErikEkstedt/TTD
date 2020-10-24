from os.path import join, exists, basename
from os import makedirs
from argparse import ArgumentParser
import subprocess
from tqdm import tqdm
import re


from research.utils import read_txt, write_json, write_txt, read_json
from research.datasets.basebuilder import BaseBuilder


def clean_persona(t):
    t = re.sub("\s([\.\!\?\,])", r"\1", t)  # remove space before punctuation
    t = re.sub("([\.\!\?\,])([\.\!\?\,])", r"\1", t)
    return t


class PersonaBuilder(BaseBuilder):
    URL = "http://parl.ai/downloads/personachat/personachat.tgz"
    NAME = "Persona"

    def download_text(self):
        if not exists(join(self.raw_data_root, "train_none_original.txt")):
            name = basename(self.URL)
            zip_name = join(self.root, name)
            if not exists(zip_name):
                print("Download data")
                # system(f"wget -q -O {zip_name} {self.URL}")
                subprocess.call(["wget", "-q", "-O", zip_name, self.URL])
            print("Extract data")
            zip_file = join(self.root, name)
            # system(f"tar -zxvf {zip_file}")
            subprocess(["tar", "-zxvf", zip_file])
            unzipped_name = join(self.root, name.replace(".tqz", ""))
            # system(f"mv personachat {self.raw_data_root}")
            subprocess(["mv", "personachat", self.raw_data_root])

    def download_audio(self):
        raise NotImplementedError(f"{self.NAME} has no audio")

    def _process_word_level(self):
        raise NotImplementedError("{self.NAME} does not contain word-level timings")

    def _process_turn_level(self):
        makedirs(self.turn_level_root, exist_ok=True)

        self.download_text()

        files = [
            "test_none_original.txt",
            "train_none_original.txt",
            "valid_none_original.txt",
        ]
        train_files, val_files, test_files = [], [], []
        dialog_num = 0

        for file in files:
            dialogs = read_txt(join(self.raw_data_root, file))
            filepaths = []
            tmp_turns = []
            turn_ind = 0
            start = 0
            for d in tqdm(dialogs, desc=file):
                n = int(d[0])
                if n > turn_ind:  # conversation continues
                    utts = d.split("\t")[:2]
                    t1 = re.sub(r"^(\d+)\s", "", utts[0])
                    t1 = clean_persona(t1)
                    tmp_turns.append({"text": t1, "speaker_id": 0, "start": start})
                    start += 1
                    t2 = clean_persona(utts[1])
                    tmp_turns.append({"text": t2, "speaker_id": 1, "start": start})
                    start += 1
                    turn_ind = n
                else:
                    # save dialog
                    filename = f"persona{dialog_num}.json"
                    write_json(tmp_turns, join(self.turn_level_root, filename))
                    filepaths.append(filename)

                    # Reset -------------------------------------------------
                    dialog_num += 1
                    tmp_turns = []
                    start = 0
                    turn_ind = n

                    # first in this dialog ----------------------------------
                    t1 = re.sub(r"^(\d+)\s", "", utts[0])
                    t1 = clean_persona(t1)
                    tmp_turns.append({"text": t1, "speaker_id": 0, "start": start})
                    start += 1
                    t2 = clean_persona(utts[1])
                    tmp_turns.append({"text": t2, "speaker_id": 1, "start": start})
                    start += 1

            if "train" in file:
                write_txt(filepaths, join(self.root, "train.txt"))
            elif "valid" in file:
                write_txt(filepaths, join(self.root, "val.txt"))
            else:
                write_txt(filepaths, join(self.root, "test.txt"))


if __name__ == "__main__":

    parser = ArgumentParser()
    parser = PersonaBuilder.add_data_specific_args(parser, name="persona")
    args = parser.parse_args()
    hparams = vars(args)
    builder = PersonaBuilder(hparams)
    builder.prepare_turn_level()

    file = join(builder.turn_level_root, builder.val_filepaths[0])
    print(read_json(file))
