from config import *

class OffsetData(Dataset):
    def __init__(self, input_size, offset, seq_length, num_samples=20000):
        self.input_size = input_size
        self.offset = offset

        self.inputs = torch.rand(num_samples, seq_length, input_size).to(DEVICE)
        self.outputs = torch.concat([torch.zeros(num_samples, offset, input_size).to(DEVICE),
                                    self.inputs[:, offset:, :]],
                                    dim=1).to(DEVICE)
    
    def __getitem__(self, index):
        return self.inputs[index].to(DEVICE), self.outputs[index].to(DEVICE)
    
    def __len__(self):
        return self.inputs.shape[0]