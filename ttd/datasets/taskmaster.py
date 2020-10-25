from argparse import ArgumentParser
from os.path import join, exists, basename
from os import makedirs
from glob import glob
import csv
import random
from tqdm import tqdm
import time

from ttd.utils import read_txt, wget, read_json, write_json, write_txt
from ttd.dialog_helpers import join_consecutive_utterances
from ttd.basebuilder import BaseBuilder


random.seed(10)  # for split creation


class TaskmasterBuilder(BaseBuilder):
    URL1 = "https://raw.githubusercontent.com/google-research-datasets/Taskmaster/master/TM-1-2019"
    URL2 = "https://github.com/google-research-datasets/Taskmaster/raw/master/TM-2-2020/data"
    URLS = {
        "TM1_self": join(URL1, "self-dialogs.json"),  # omit the self-dialogs
        "TM1_woz": join(URL1, "woz-dialogs.json"),
        "TM2_flights": join(URL2, "flights.json"),
        "TM2_food_ordering": join(URL2, "food-ordering.json"),
        "TM2_hotels": join(URL2, "hotels.json"),
        "TM2_movies": join(URL2, "movies.json"),
        "TM2_music": join(URL2, "music.json"),
        "TM2_restaurant_search": join(URL2, "restaurant-search.json"),
        "TM2_sports": join(URL2, "sports.json"),
    }
    SPLIT = {"train": 0.9, "test": 0.05}  # validation is remaining
    tm1_train = join(URL1, "train-dev-test/train.csv")
    tm1_test = join(URL1, "train-dev-test/test.csv")
    tm1_dev = join(URL1, "train-dev-test/dev.csv")
    NAME = "Taskmaster"

    def download_text(self):
        """ Downloads files to {root}/raw_data """
        makedirs(self.raw_data_root, exist_ok=True)
        if not exists(join(self.raw_data_root, "tm1_train.txt")):
            for name, url in self.URLS.items():
                dest = join(self.raw_data_root, name + ".json")
                print("#" * 70)
                print(f"Downloading {name}...")
                if not exists(dest):
                    wget(url, dest)
                print("#" * 70)
                if name == "TM1_self":
                    wget(self.tm1_train, join(self.raw_data_root, "tm1_train.txt"))
                    wget(self.tm1_dev, join(self.raw_data_root, "tm1_dev.txt"))
                    wget(self.tm1_test, join(self.raw_data_root, "tm1_test.txt"))

    def download_audio(self):
        raise NotImplementedError("Taskmaster has no audio")

    def _process_turn_level(self):
        makedirs(self.turn_level_root, exist_ok=True)

        self.download_text()

        train_filepaths = []
        val_filepaths = []
        test_filepaths = []

        total, skipped = 0, 0
        t = time.time()
        for json_path in tqdm(glob(join(self.raw_data_root, "*.json")), desc=self.NAME):
            data_name = basename(json_path).replace(".json", "")
            dialogs = read_json(json_path)

            if data_name == "TM1_self":
                self_train = read_txt(join(self.raw_data_root, "tm1_train.txt"))
                self_val = read_txt(join(self.raw_data_root, "tm1_dev.txt"))
                self_test = read_txt(join(self.raw_data_root, "tm1_test.txt"))

                # clean comma (originally a csv file)
                self_train = [f.strip(",") for f in self_train]
                self_val = [f.strip(",") for f in self_val]
                self_test = [f.strip(",") for f in self_test]

            filenames = []
            for dialog_data in dialogs:
                filename = "-".join(
                    [
                        data_name,
                        dialog_data["conversation_id"],
                        dialog_data["instruction_id"],
                    ]
                )
                # _ is used when concatenating dsets
                filename = filename.replace("_", "-")
                filename += ".json"
                # filename too long?
                dialog = self._extract_turn_level_dialogs(dialog_data)

                if dialog is None or len(dialog) < 2:
                    skipped += 1
                else:
                    dialog = join_consecutive_utterances(dialog)
                    if len(dialog) > 1:
                        write_json(dialog, join(self.turn_level_root, filename))
                        total += 1
                        # tm1_self_dialogs contain predefined train/dev/test splits
                        # using dev as val.
                        if data_name == "TM1_self":
                            if dialog_data["conversation_id"] in self_train:
                                train_filepaths.append(filename)
                            elif dialog_data["conversation_id"] in self_val:
                                val_filepaths.append(filename)
                            else:  # test
                                test_filepaths.append(filename)
                        else:
                            filenames.append(filename)
            # create splits for each data-group (restaurants, hotels, etc)
            if data_name != "TM1_self":
                train, val, test = self._create_splits(filenames)
                train_filepaths += train
                val_filepaths += val
                test_filepaths += test
        t = time.time() - t
        print(f"Preprocessing took {round(t, 1)} seconds.")
        print("Skipped: ", skipped)
        print("Total dialogs: ", total)
        print("Train: ", len(train_filepaths))
        print("Val: ", len(val_filepaths))
        print("Test: ", len(test_filepaths))
        write_txt(train_filepaths, join(self.root, "train.txt"))
        write_txt(val_filepaths, join(self.root, "val.txt"))
        write_txt(test_filepaths, join(self.root, "test.txt"))

    def _extract_turn_level_dialogs(self, dialog_data):
        conversation = []
        for utt in dialog_data["utterances"]:
            if utt["speaker"].lower() == "assistant":
                speaker_id = 0
            elif utt["speaker"].lower() == "user":
                speaker_id = 1
            else:
                return None  # do not know the speaker, omit file
            conversation.append(
                {
                    "text": utt["text"],
                    "speaker_id": speaker_id,
                    "start": utt["index"],
                }
            )
        conversation.sort(key=lambda x: x["start"])
        return conversation

    def _create_splits(self, filenames):
        random.shuffle(filenames)
        n_train = int(len(filenames) * self.SPLIT["train"])
        n_test = int(len(filenames) * self.SPLIT["test"])
        n_val = len(filenames) - n_train - n_test
        train = filenames[:n_train]
        test = filenames[n_train : n_train + n_test]
        val = filenames[n_train + n_test :]
        return train, val, test

    def _process_word_level(self):
        raise NotImplementedError("{self.NAME} does not contain word-level timings")


if __name__ == "__main__":

    parser = ArgumentParser()
    parser = TaskmasterBuilder.add_data_specific_args(parser, name="taskmaster")
    args = parser.parse_args()
    hparams = vars(args)
    builder = TaskmasterBuilder(hparams)
    builder.prepare_turn_level()

    file = join(builder.turn_level_root, builder.val_filepaths[0])
    print(read_json(file))
