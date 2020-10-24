from os.path import join, exists, basename
from os import makedirs
import subprocess
from argparse import ArgumentParser
from tqdm import tqdm


from research.utils import write_txt, read_txt, wget, read_json, write_json
from research.datasets.basebuilder import BaseBuilder


class MultiwozBuilder(BaseBuilder):
    URL = "https://www.repository.cam.ac.uk/bitstream/handle/1810/280608/MULTIWOZ2.zip?sequence=3&isAllowed=y"
    NAME = "MultiWOZ"

    def _process_word_level(self):
        raise NotImplementedError("{self.NAME} does not contain word-level timings")

    def download_audio(self):
        raise NotImplementedError("Taskmaster has no audio")

    def download_text(self):
        to = join(self.root, self.NAME.lower() + ".zip")
        if not exists(self.raw_data_root):
            if not exists(to):
                print("Download")
                wget(self.URL, to)
            subprocess.call(["unzip", to, "-d", self.root])
            subprocess.call(["mv", join(self.root, "MULTIWOZ2 2"), self.raw_data_root])

            macos_dir = join(self.root, "__MACOSX")
            if exists(macos_dir):
                subprocess.call(["rm", "-r", macos_dir])

    def _process_turn_level(self):
        makedirs(self.turn_level_root, exist_ok=True)

        self.download_text()  # downloads if necessary

        data = read_json(join(self.raw_data_root, "data.json"))
        test_filepaths = read_txt(join(self.raw_data_root, "testListFile.json"))
        val_filepaths = read_txt(join(self.raw_data_root, "valListFile.json"))
        train_filepaths = []

        for session_name, v in tqdm(data.items(), desc=f"{self.NAME} Turn-level"):
            dialog = []
            start = 0
            for i, utt in enumerate(v["log"]):
                speaker_id = 0 if i % 2 == 0 else 1
                dialog.append(
                    {
                        "text": utt["text"],
                        "speaker_id": speaker_id,
                        "start": start,
                    }
                )
                start += 1

            # we only know which files are for validation and testing
            if not (session_name in test_filepaths or session_name in val_filepaths):
                train_filepaths.append(session_name)

            # save file
            write_json(dialog, join(self.turn_level_root, session_name))
        write_txt(train_filepaths, join(self.root, "train.txt"))
        write_txt(val_filepaths, join(self.root, "val.txt"))
        write_txt(test_filepaths, join(self.root, "test.txt"))


if __name__ == "__main__":

    parser = ArgumentParser()
    parser = MultiwozBuilder.add_data_specific_args(parser, name="multiwoz")
    args = parser.parse_args()
    hparams = vars(args)
    builder = MultiwozBuilder(hparams)
    builder.prepare_turn_level()

    file = join(builder.turn_level_root, builder.val_filepaths[0])
    print(read_json(file))
