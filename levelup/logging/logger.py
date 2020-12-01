""" Log generic training data. """
import os
import pandas as pd


class Logger:
    def __init__(self):
        self.data: pd.DataFrame = pd.DataFrame()

    def reset(self):
        self.data = pd.DataFrame()

    def log(self, name: str, epoch: int, episode: int, reward: float):
        self.data = self.data.append({"name": name,
                                      "epoch": epoch,
                                      "episode": episode,
                                      "reward": reward}, ignore_index=True)

    def write(self, file_path: str):
        if os.path.exists(file_path):
            self.data.to_csv(file_path, mode='a', header=False)
        else:
            self.data.to_csv(file_path)
        self.reset()
