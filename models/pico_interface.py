import struct
import torch
from torch import nn

from dataset import FPDataset
from models.base import BaseModel

from serial import Serial

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)


def pack_input(eval_flag, inputs):
    return struct.pack('<B36f', eval_flag, *inputs)

def unpack_output(response_bytes):
    return struct.unpack('<2f', response_bytes)

class NormalizeInput(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x / (x.norm(dim=1, keepdim=True) + 1e-8)

"""
Pico Interface for Serial Communication
This class interfaces with a Pico device over serial communication to send input data and receive predictions.
"""
class PicoInterface(BaseModel):
    def __init__(self, serial_port='/dev/ttyACM0'):
        """
        Initialize the Pico Interface.

        :param serial_port: The serial port to use for communication.
        """
        self.serial = Serial(serial_port, 9600, timeout=1)

    def fit(self, dataset: FPDataset):
        """
        Fit the model to the dataset.

        :param dataset: The dataset to fit the model to.
        """
        pass

    def predict(self, X: torch.Tensor, eval: bool=False) -> torch.Tensor:
        """
        Predict using the model on the dataset.

        :param X: The data to predict on.
        :param eval: Whether or not it's in evaluation mode.
        :return: The predictions.
        """
        
        output = []
        # If eval is True, we send all inputs at once and read the scalar output in a loop.
        if eval:
            for i in range(0, len(X)):
                X_i = X[i].unsqueeze(0)
                self.serial.write(pack_input(eval, X_i.flatten().tolist()))
            
            for _ in range(18):
                response_bytes = self.serial.read_until(b'BIGGER_THAN_8_BYTES', 8)
                x, y = unpack_output(response_bytes)
                output.append(x)
                output.append(y)
            
            return torch.tensor(output, dtype=torch.float32)

        # If eval is False, we send each input one by one and read the prediction output.
        for i in range(0, len(X)):
            X_i = X[i].unsqueeze(0)
            self.serial.write(pack_input(eval, X_i.flatten().tolist()))
            response_bytes = self.serial.read_until(b'BIGGER_THAN_8_BYTES', 8)
            x, y = unpack_output(response_bytes)
            output.append([x, y])

        return torch.tensor(output, dtype=torch.float32)

    def save(self, model_path: str):
        """
        Save the model to the specified path using pickle.

        :param model_path: The path to save the model.
        """
        pass

    def load(self, model_path: str):
        """
        Load the model from the specified path using pickle.

        :param model_path: The path to load the model from.
        :return: The loaded model.
        """
        pass
