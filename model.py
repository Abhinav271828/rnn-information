from data import *

class ToyModels(pl.LightningModule):
    def __init__(self, input_size, hidden_size, offset, lr, lf):
        super().__init__()
        self.save_hyperparameters(ignore=['lf'])
        self.is_ = input_size
        self.hs = hidden_size
        self.offset = offset
        self.lr = lr
        self.lf = lf
    
    def get_dataloader(self, seq_length=200, num_samples=20000):
        ds = OffsetData(self.is_, self.offset, seq_length, num_samples)
        dl = DataLoader(ds, batch_size=BATCH_SIZE, num_workers=7, persistent_workers=True)
        return dl

    def step(self, loss_type, batch, batch_idx):
        hs, pred = self(batch[0])
        loss = self.lf(pred.to(DEVICE), batch[1].to(DEVICE))
        self.log(loss_type, loss, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self.step("train_loss", batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        return self.step("val_loss", batch, batch_idx)
    
    def test_step(self, batch, batch_idx):
        return self.step("test_loss", batch, batch_idx)

    def train_dataloader(self):
        return self.get_dataloader()

    def val_dataloader(self):
        return self.get_dataloader(num_samples=5000)
    
    def test_dataloader(self):
        ds = torch.load(f'testing_dataset-in={self.is_}-offset={self.offset}.pkl', map_location=DEVICE)
        dl = DataLoader(ds, batch_size=BATCH_SIZE)
        return dl

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

class ToyRNN(ToyModels):
    def __init__(self, input_size, hidden_size, offset, lr=0.1, lf=nn.MSELoss(), nonlinearity='relu', bias=True):
        super().__init__(input_size, hidden_size, offset, lr=lr, lf=lf)
        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size,
                          batch_first=True, nonlinearity=nonlinearity, bias=bias)
        self.ffn = nn.Sequential(nn.Linear(in_features=hidden_size, out_features=input_size, bias=bias),
                                 nn.ReLU())
        self.model_type = "nonlin"
        self.to(DEVICE)

    def forward(self, batch):
                      # [bz, seq, is]
        batch = batch.to(DEVICE)
        self = self.to(DEVICE)
        hidden_states, _ = self.rnn(batch)
        # [bz, seq, hs]
        preds = self.ffn(hidden_states)
        # [bz, seq, is]
        return hidden_states, preds
