from abc import ABC, abstractmethod
from typing import Tuple

class Callback(ABC):
    def __init__(self, file_path: str, overwrite: bool = False):
        """
        Initialize the callback with a file path and an overwrite flag.
        """
        self._file_path = file_path
        if os.path.isfile(file_path) and not overwrite:
            raise ValueError(f"File {file_path} already exists. Set overwrite=True to overwrite.")
        else:
            # Create the file and write the header.
            with open(file_path, 'w') as f:
                f.write("epoch,loss\n")
        
        def on_epoch_end(self, epoch: int, loss: float):
            """
            Called at the end of each epoch.
            """
            with open(self._file_path, 'a') as f:
                f.write(f"{epoch},{loss}\n") #write the epoch and loss to the file.