from argparse import ArgumentParser
from os.path import join, exists, basename, isdir
from os import makedirs, listdir
import subprocess
import shutil
from glob import glob
from tqdm import tqdm
import re

import torch

from ttd.basebuilder import BaseBuilder
from ttd.dialog_helpers import word_level_to_turns
from ttd.utils import (
    read_json,
    read_txt,
    write_json,
    get_duration_sox,
    get_sample_rate_sox,
)


SWB_OMIT = [
    "[silence]",
    "[noise]",
    "[laughter]",
    "[vocalized-noise]",
    "[noise]",
    "<b_aside>",
    "<e_aside>",
]


SWB_WORD_MAP = {
    "uh-huh": "uhuh",
    "huh-uh": "uhuh",
    "uh-hum": "mhm",
    "uh-hums": "mhm",
    "um-hum": "mhm",
    "hum-um": "mhm",
    "uh-oh": "uhoh",
}


def clean_swb_word(w):
    """
    Guide:

    https://www.isip.piconepress.com/projects/switchboard/doc/transcription_guidelines/transcription_guidelines.pdf

    15. Partial words:  w[ent], the spoken part "w" and the unspoken part ["ent"] -> went
    16. Restarts of i: "i- i really do" -> use the spoken restarts: "i i really do"
    17. Mispronounciation [bocket/bucket] = [spoken_word/actual_word] -> actual_word e.g. bucket
    18. Coinage: new words {weatherwise}  -> weatherwise
    19. Asides: sometimes the interlocutors speaks to someone else (e.g. speaking with family)
        - <b_side>, <e_side> marks the beginning and end of words not "really part of" the conversation
        - we just omit these tags and keep the spoken words (even if they are meant for someone else)
    20. Hesitation sound: "ah", "uh" (vowel sound) and "um", "hm" (nasal sound)
    21. Yes/no sound: Use "uh-huh" or "um-hum" (yes) and "huh-uh" or "hum-um" (no)
    22. Non-speech: [laughter], [noise], [vocalized-noise] -> skip
    23. Laughter during speech: [laughter-yes] -> yes
    24. Pronounciation variants: about_1, okay_1, etc  okay_1 = mkay -> use the word e.g. okay
    26. Special lexical issues: "alright" is transcribed as "all right" -> split into two words
    """
    w = SWB_WORD_MAP.get(w, w)
    w = re.sub(r"\[laughter\-", "", w)  # laughing and speaker e.g. [laughter-yeah]
    w = re.sub("\]", "", w)  # laughing and speaker e.g. [laughter-yeah]
    # w = re.sub(r"\[vocalized\-*\w*\]", "", w)
    w = re.sub(r"\[.*\/(\w\w*)\]", r"\1", w)  # 17. mispronounced words
    w = re.sub(r".*\/(\w\w*)", r"\1", w)  # 17. mispronounced words missing [] ?
    w = re.sub(r"_1", "", w)  # 24. pronounciation variants
    w = re.sub(r"\[", "", w)
    w = re.sub(r"\]", "", w)
    w = re.sub(r"\{(.*)\}", r"\1", w)
    w = re.sub(r"\-+$", "", w)  # 16. restarts
    w = re.sub(r"^\-+", "", w)  #
    w = SWB_WORD_MAP.get(w, w)
    return w


def check_swb_words():
    from research.datasets.verbal import WordDataset

    parser = ArgumentParser()
    parser = SwitchboardDataModuleBase.add_data_specific_args(parser)
    args = parser.parse_args()
    dm = SwitchboardDataModuleBase(vars(args))
    dm.prepare_data()
    dset = WordDataset(dm.dialog_root, clean_function=dm.clean_word_function)
    non_alpha = set()
    for words, speaker_id, starts, name in tqdm(dset):
        for w in words:
            w = re.sub(r"\'", "", w)
            if not w.isalpha():
                non_alpha.add(w)
    print(len(non_alpha))
    return non_alpha


class SwitchboardBuilder(BaseBuilder):
    URL = "https://www.isip.piconepress.com/projects/switchboard/releases/switchboard_word_alignments.tar.gz"
    N_FILES = 2438
    NAME = "Switchboard"
    AUDIO_EXT = ".sph"
    SAMPLE_RATE = 8000

    def download_text(self):
        """Downloads switchboard annotations

         self.root
        └──  swb_ms98_transcriptions
           ├──  20
           │  ├──  2001
           │  ├── ...
           ├──  49
           │  ├── ...
           │  └──  4940
           ├──  AAREADME.text
           └──  sw-ms98-dict.text
        """
        tar_path = join(self.root, basename(self.URL))
        print(f"Downloading {self.NAME} annotations")
        if not exists(tar_path):
            subprocess.call(
                ["wget", "-P", self.root, self.URL, "-q", "--show-progress"]
            )

        print("Extracting")
        subprocess.call(["tar", "xzf", tar_path, "-C", self.root])
        shutil.move(
            join(self.root, "swb_ms98_transcriptions"), join(self.raw_data_root)
        )
        print("Annotations download complete!")

    def download_audio(self):
        raise NotImplementedError(
            "Not freely available... Can't download switchboard audio"
        )

    def _process_turn_level(self):
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
            )

            write_json(word_level_turns, join(self.turn_level_root, json_name))

    def _process_word_level(self):
        print(f"{self.NAME}: process_word_level")  # logger

        # check if word level exits
        if not self.check_if_dir_exists(self.word_level_root):
            print(f"{self.NAME}: world level data not found")
            if not (exists(self.raw_data_root) and isdir(self.raw_data_root)):
                print("raw data not found: ", self.raw_data_root)
                self.download_text()

            # Extract wordlevel
            makedirs(self.word_level_root, exist_ok=True)
            omitted_words = 0
            changed_words = 0
            total_words = 0
            for DD in tqdm(
                listdir(self.raw_data_root), desc=f"{self.NAME} Word level dialogs"
            ):
                DD = join(self.raw_data_root, DD)
                if isdir(DD):
                    for dialog_number in listdir(DD):
                        dialog_id = f"sw{dialog_number}"
                        A_words = read_txt(
                            join(DD, dialog_number, dialog_id + "A-ms98-a-word.text")
                        )  # speaker A
                        B_words = read_txt(
                            join(DD, dialog_number, dialog_id + "B-ms98-a-word.text")
                        )  # speaker B
                        # A/B_words is a list of strings
                        # each string:
                        #   'sw3856A-ms98-a-0002 0.925625 1.233875 what'
                        #   '{dialog_id}{speaker_id}-ms98-{utt_id} {start} {end} {word}'
                        #   dialog_id = sw3856
                        #   speaker_id = a-0002
                        #   start = 0.925625
                        #   end =  1.233875
                        #   word =  what
                        dialog_words = []
                        for speaker_id, word_list in enumerate([A_words, B_words]):
                            for word_data in word_list:
                                id, start, end, word = word_data.split()

                                if word == "[silence]":
                                    continue

                                total_words += 1

                                if word in SWB_OMIT:
                                    omitted_words += 1
                                    continue

                                w = clean_swb_word(word)
                                if w != word:
                                    changed_words += 1

                                start = float(start)
                                end = float(end)
                                utt_id = id.split("ms98-")[-1]  # utterance id

                                dialog_words.append(
                                    {
                                        "word": w,
                                        "start": start,
                                        "end": end,
                                        "utt_id": utt_id,
                                        "speaker_id": speaker_id,  # 0 or 1
                                    }
                                )
                        # sort words by start
                        dialog_words = [
                            dw
                            for dw in sorted(
                                dialog_words, key=lambda item: item["start"]
                            )
                        ]

                        write_json(
                            dialog_words,
                            join(self.word_level_root, dialog_id + ".json"),
                        )

            print(
                f"Omitted {omitted_words} {round(100 * omitted_words / total_words, 3)}% of words"
            )
            print(
                f"Changed {changed_words} {round(100 * changed_words / total_words, 3)}% of words"
            )
            print("-" * 50)


if __name__ == "__main__":

    parser = ArgumentParser()
    parser = SwitchboardBuilder.add_data_specific_args(parser, name="switchboard")
    args = parser.parse_args()
    hparams = vars(args)
    builder = SwitchboardBuilder(hparams)
    builder.prepare_turn_level()

    file = join(builder.turn_level_root, builder.val_filepaths[0])
    print(read_json(file))
