from argparse import ArgumentParser
from os.path import join, exists, basename
from os import makedirs
import subprocess
from glob import glob
from tqdm import tqdm
import random
import time
import json


from ttd.utils import write_txt, read_txt, wget, read_json, write_json
from ttd.basebuilder import BaseBuilder


random.seed(10)  # for split creation


class MetalwozBuilder(BaseBuilder):
    URL = "https://download.microsoft.com/download/E/B/8/EB84CB1A-D57D-455F-B905-3ABDE80404E5/metalwoz-v1.zip"
    NAME = "MetaLWOZ"
    SPLIT = {"train": 0.9, "test": 0.05}  # validation is remaining

    def _process_word_level(self):
        raise NotImplementedError("{self.NAME} does not contain word-level timings")

    def download_audio(self):
        raise NotImplementedError("Taskmaster has no audio")

    def download_text(self):
        if not exists(self.raw_data_root):
            zip_name = join(self.root, basename(self.URL))
            if not exists(zip_name):
                print("Download")
                wget(self.URL, zip_name)

            subprocess.call(["unzip", zip_name, "-d", self.root])
            subprocess.call(["mv", join(self.root, "dialogues"), self.raw_data_root])
            subprocess.call(["mv", join(self.root, "LICENSE.pdf"), self.raw_data_root])
            subprocess.call(["mv", join(self.root, "tasks.txt"), self.raw_data_root])

    def _process_turn_level(self):
        """
        There is no pre-defined splits so we set random seed (see imports) and split all domains into train/val/test

        Dialog data looks like this:
            string-data = '{
                    "id": "af15eaa8",
                    "user_id": "19b006ed",
                    "bot_id": "3f60b0cb",
                    "domain": "TIME_ZONE",
                    "task_id": "c60e80fc",
                    "turns": ["Hello how may I help you?",
                        "Hi there,
                        could you explain to me how time zones work? I don't really understand it",
                        "Hey! I am only able to calculate times in diffferent tome zones.",
                        "Oh,
                        so you can't explain how they work,
                        you can only find out local times?",
                        "Correct",
                        "Alright,
                        could you tell me what time it is in Los Angeles now?",
                        "It is currently 1:25am in Los Angeles",
                        "Great! And do you know if Sacramento is in the same time zone?",
                        "It is indeed",
                        "Alright,
                        thanks for the info",
                        "Happy to help!"]
                    }'
        """
        makedirs(self.turn_level_root, exist_ok=True)

        self.download_text()  # downloads if necessary

        train_filepaths = []
        val_filepaths = []
        test_filepaths = []
        total, skipped = 0, 0

        t = time.time()
        for datafile in tqdm(
            glob(join(self.raw_data_root, "*.txt")), desc=f"{self.NAME} Turn-level"
        ):
            filename = basename(datafile)
            if filename == "tasks.txt":
                continue

            filenames = []
            data = read_txt(datafile)
            for string_data in data:
                dict_data = json.loads(string_data)
                # dict_data.keys(): ['id', 'user_id', 'bot_id', 'domain', 'task_id', 'turns']
                conversation = []
                for i, utt in enumerate(dict_data["turns"]):
                    if i % 2 == 0:
                        speaker_id = 0
                    else:
                        speaker_id = 1
                    conversation.append(
                        {
                            "text": utt,
                            "speaker_id": speaker_id,
                            "start": i,
                        }
                    )

                if len(conversation) > 1:
                    savename = "-".join([dict_data["domain"], dict_data["id"]])
                    savename += ".json"
                    write_json(conversation, join(self.turn_level_root, savename))
                    filenames.append(savename)
                    total += 1
                else:
                    skipped += 1
            # create splits for each domain
            train, val, test = self._create_splits(filenames)
            train_filepaths += train
            val_filepaths += val
            test_filepaths += test
        t = time.time() - t
        print(self.NAME)
        print(f"Preprocessing took {round(t, 1)} seconds.")
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
    parser = MetalwozBuilder.add_data_specific_args(parser, name="metalwoz")
    args = parser.parse_args()
    hparams = vars(args)
    builder = MetalwozBuilder(hparams)
    builder.prepare_turn_level()

    file = join(builder.turn_level_root, builder.val_filepaths[0])
    print(read_json(file))
