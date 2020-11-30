from argparse import ArgumentParser
from os.path import join, split, basename, exists, isdir
from os import makedirs, listdir
import subprocess
import numpy as np
import xml.etree.ElementTree as ET
import shutil
from tqdm import tqdm
from glob import glob
import re
import torch

from ttd.basebuilder import BaseBuilder
from ttd.dialog_helpers import word_level_to_turns
from ttd.utils import (
    read_json,
    write_json,
    get_duration_sox,
    get_sample_rate_sox,
)


import time


MAPTASK_WORD_MAP = {
    "mm-hmm": "mhm",
    "mm-hm": "mhm",
    "mmhmm": "mhm",
    "mm-mm": "mmm",
    "uh-huh": "uhuh",
    "uh-uh": "uhuh",
    "right-o": "righto",
}


def clean_maptask_word(w):
    w = MAPTASK_WORD_MAP.get(w, w)
    w = re.sub('"', "", w)  # remove quotations
    w = re.sub("^'", "", w)  # remove "'" at the start ("'bout", "'cause")
    w = re.sub("'$", "", w)  # remove "'" at the end ("apaches'")
    w = re.sub("\-\-+", "", w)
    w = re.sub("\-\w(\-\w)+", "", w)  # na-a-a-a-a -> na
    return w


class MaptaskBuilder(BaseBuilder):
    URL = "http://groups.inf.ed.ac.uk/maptask/hcrcmaptask.nxtformatv2-1.zip"
    AUDIO_URL = "http://groups.inf.ed.ac.uk/maptask/signals/dialogues"
    NAME = "Maptask"
    N_FILES = 128
    AUDIO_EXT = ".wav"
    SAMPLE_RATE = 20000

    def download_text(self):
        """
        Downloads maptask annotations
        """
        tmp_root = "/tmp"
        wget_cmd = ["wget", "-P", tmp_root, self.URL, "-q", "--show-progress"]

        print(f"Downloading {self.NAME} annotations")
        print("-----------------------")
        subprocess.call(wget_cmd)
        print("Download complete")

        zip_path = join(tmp_root, "hcrcmaptask.nxtformatv2-1.zip")
        print("download: ", zip_path)

        subprocess.call(["unzip", "-qq", zip_path, "-d", tmp_root])
        shutil.move(join(tmp_root, "maptaskv2-1"), self.raw_data_root)
        subprocess.call(["rm", zip_path])

    def download_audio(self):
        """
        Downloads maptask audio files into:
            research/research/datasets/$ROOT/MapTask/audio
        """
        wget_cmd = [
            "wget",
            "-P",
            self.audio_root,
            "-r",
            "-np",
            "-R",
            "index.html*",
            "-nd",
            self.AUDIO_URL,
            "-q",
            "--show-progress",
        ]
        try:
            subprocess.call(wget_cmd)
        except:
            print("Download interrupted...")

        # rename audio files: q1ec1.mix.wav -> q1ec1.wav
        for wav_file in glob(join(self.audio_root, "*.mix.wav")):
            shutil.move(wav_file, wav_file.replace(".mix", ""))

        for f in listdir(self.audio_root):
            fpath = join(self.audio_root, f)
            if not f.endswith(".wav"):
                subprocess.call(["rm", fpath])

    def _process_turn_level(self):
        """
        The super class contains higher level functions used by diffent dataset such as "prepare_turn_level".

        Theses prepare classes checks if the files exists but if they do not it calls the dataset specific
        '_process_turn_level' which extract the relevant data.
        """
        print(f"{self.NAME}: process_turn_level (slow)")

        # From super class. makes sure that the data we need exists
        self.prepare_word_level()  # word-level-dialog required
        self.prepare_vad()  # processed vad values required

        # Extract Turn level
        makedirs(self.turn_level_root, exist_ok=True)

        # loop over entries in the word-level processing and transform to turns
        word_level_files = glob(join(self.word_level_root, "*.json"))
        for word_level_path in tqdm(word_level_files):
            json_name = basename(word_level_path)

            audio_path = self.get_audio_path(json_name.replace(".json", ""))
            vad_path = join(self.vad_root, json_name.replace(".json", ".pt"))

            word_level_dialog = read_json(word_level_path)
            vad = torch.load(vad_path)  # list of (start, end) times
            duration = get_duration_sox(audio_path)
            sr = get_sample_rate_sox(audio_path)

            word_level_turns = word_level_to_turns(
                word_level_dialog,
                vad,
                duration,
                sr,
                # vad_step_time=vad_step_time,
                # vad_pad=vad_pad,
                # ipu_thresh=ipu_thresh,
            )

            write_json(word_level_turns, join(self.turn_level_root, json_name))

    def _process_word_level(self):
        print(f"{self.NAME}: process_word_level")  # logger

        # check if word level exits
        print(self.word_level_root)
        if not self.check_if_dir_exists(self.word_level_root):
            print("world level data not found")
            if not (exists(self.raw_data_root) and isdir(self.raw_data_root)):
                print("raw data not found: ", self.raw_data_root)
                self.download_text()

            # Extract wordlevel
            makedirs(self.word_level_root, exist_ok=True)

            tu_path = join(self.raw_data_root, "Data/timed-units")
            pos_path = join(self.raw_data_root, "Data/pos")
            token_path = join(self.raw_data_root, "Data/tokens")

            dialog_ids = set([f.split(".")[0] for f in listdir(tu_path)])
            for dialog_id in tqdm(dialog_ids, desc=f"{self.NAME} Dialogs"):
                tu_path_g = join(tu_path, dialog_id + ".g.timed-units.xml")
                tu_path_f = join(tu_path, dialog_id + ".f.timed-units.xml")

                dialog_words = self._extract_words(tu_path_g, speaker_id=0)
                dialog_words += self._extract_words(tu_path_f, speaker_id=1)
                dialog_words.sort(key=lambda x: x["start"])
                write_json(
                    dialog_words, join(self.word_level_root, dialog_id + ".json")
                )

    def _extract_words(self, xml_path, speaker_id):
        xml_element_tree = ET.parse(xml_path)
        all_words = []
        for elem in xml_element_tree.iter():
            try:
                start = float(elem.attrib["start"])
                end = float(elem.attrib["end"])
            except:
                continue

            if elem.tag == "tu":
                # elem.attrib: start, end, utt
                # tu.append({"time": tmp, "words": elem.text})
                word = elem.text
                list_words = self._format_word_entry(
                    word=elem.text,
                    start=start,
                    end=end,
                    idd=elem.attrib["id"],
                    speaker_id=speaker_id,
                )
                all_words += list_words
        all_words.sort(key=lambda x: x["start"])
        return all_words

    def _format_word_entry(self, word, start, end, idd, speaker_id):
        clean_dialog = []
        if " " in word:  # multiple words in one
            for word in word.split():
                word = clean_maptask_word(word)
                clean_dialog.append(
                    {
                        "word": word,
                        "start": start,
                        "end": end,
                        "id": idd,
                        "speaker_id": speaker_id,
                    }
                )
        else:
            word = clean_maptask_word(word)
            words = word.split("-")
            if len(words) > 1:  # right-hand, left-hand- s-shaped
                for word in words:
                    word = clean_maptask_word(word)
                    clean_dialog.append(
                        {
                            "word": word,
                            "start": start,
                            "end": end,
                            "id": idd,
                            "speaker_id": speaker_id,
                        }
                    )
            else:
                clean_dialog.append(
                    {
                        "word": words[0],
                        "start": start,
                        "end": end,
                        "id": idd,
                        "speaker_id": speaker_id,
                    }
                )
        return clean_dialog


if __name__ == "__main__":

    parser = ArgumentParser()
    parser = MaptaskBuilder.add_data_specific_args(parser, name="maptask")
    args = parser.parse_args()
    hparams = vars(args)
    builder = MaptaskBuilder(hparams)
    builder.prepare_turn_level()

    file = join(builder.turn_level_root, builder.val_filepaths[0])
    print(read_json(file))
