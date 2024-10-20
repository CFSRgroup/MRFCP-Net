import torch
import torch.nn as nn

class TransformerEncoders1(nn.Module):
    def __init__(self):
        super(TransformerEncoders1, self).__init__()
        N = [0, 12, 4, 2, 4, 11]
        self.N = N
        self.encoder_pd = nn.TransformerEncoderLayer(
            d_model=N[1],
            nhead=1,
            dim_feedforward=32,
            batch_first=True
        )
        self.liner_pd = nn.Linear(
            in_features=N[1],
            out_features= 64

        )

        self.encoder_di = nn.TransformerEncoderLayer(
            d_model=N[2],
            nhead=1,
            dim_feedforward=32,
            batch_first=True
        )
        self.liner_di = nn.Linear(
            in_features=N[2],
            out_features= 64

        )

        self.encoder_fd = nn.TransformerEncoderLayer(
            d_model=N[3],
            nhead=1,
            dim_feedforward=32,
            batch_first=True
        )
        self.liner_fd = nn.Linear(
            in_features=N[3],
            out_features= 64

        )

        self.encoder_sa = nn.TransformerEncoderLayer(
            d_model=N[4],
            nhead=1,
            dim_feedforward=32,
            batch_first=True
        )
        self.liner_sa = nn.Linear(
            in_features=N[4],
            out_features= 64

        )

        self.encoder_es = nn.TransformerEncoderLayer(
            d_model=N[5],
            nhead=1,
            dim_feedforward=32,
            batch_first=True
        )
        self.liner_es = nn.Linear(
            in_features=N[5],
            out_features= 64

        )


    def forward(self, x_eye):
        batch, sequence, _ = x_eye.shape

        x_pd = self.encoder_pd(x_eye[:, :, 0:12])
        x_di = self.encoder_di(x_eye[:, :, 12:16])
        x_fd = self.encoder_fd(x_eye[:, :, 16:18])
        x_sa = self.encoder_sa(x_eye[:, :, 18:22])
        x_es = self.encoder_es(x_eye[:, :, 22:33])

        x_pd_all = []
        x_di_all = []
        x_fd_all = []
        x_sa_all = []
        x_es_all = []

        for i in range(0, sequence):
            x_pd_out = self.liner_pd(x_pd[:, i, :].squeeze())
            x_di_out = self.liner_di(x_di[:, i, :].squeeze())
            x_fd_out = self.liner_fd(x_fd[:, i, :].squeeze())
            x_sa_out = self.liner_sa(x_sa[:, i, :].squeeze())
            x_es_out = self.liner_es(x_es[:, i, :].squeeze())

            x_pd_all.append(x_pd_out)
            x_di_all.append(x_di_out)
            x_fd_all.append(x_fd_out)
            x_sa_all.append(x_sa_out)
            x_es_all.append(x_es_out)

        x_pd = torch.stack(x_pd_all,dim=1)
        x_di = torch.stack(x_di_all,dim=1)
        x_fd = torch.stack(x_fd_all,dim=1)
        x_sa = torch.stack(x_sa_all,dim=1)
        x_es = torch.stack(x_es_all,dim=1)

        x_all = torch.stack([x_pd, x_di, x_fd, x_sa, x_es], dim=2)

        return x_all 


class TransformerEncoders2(nn.Module):
    def __init__(self):
        super(TransformerEncoders2, self).__init__()
        N = [0, 12, 4, 2, 4, 11]
        self.N = N
        self.encoder_pd = nn.TransformerEncoderLayer(
            d_model=N[1],
            nhead=1,
            dim_feedforward=32,
            batch_first=True
        )
        self.encoder_di = nn.TransformerEncoderLayer(
            d_model=N[2],
            nhead=1,
            dim_feedforward=32,
            batch_first=True
        )
        self.encoder_fd = nn.TransformerEncoderLayer(
            d_model=N[3],
            nhead=1,
            dim_feedforward=32,
            batch_first=True
        )
        self.encoder_sa = nn.TransformerEncoderLayer(
            d_model=N[4],
            nhead=1,
            dim_feedforward=32,
            batch_first=True
        )
        self.encoder_es = nn.TransformerEncoderLayer(
            d_model=N[5],
            nhead=1,
            dim_feedforward=32,
            batch_first=True
        )

        self.liner = nn.Sequential(
                   nn.Linear(in_features=33, out_features=10),
                   nn.BatchNorm1d(10)
                   )


    def forward(self, x_eye):

        x_pd = self.encoder_pd(x_eye[:,:,0:12])
        x_di = self.encoder_di(x_eye[:,:,12:16])
        x_fd = self.encoder_fd(x_eye[:,:,16:18])
        x_sa = self.encoder_sa(x_eye[:,:,18:22])
        x_es = self.encoder_es(x_eye[:,:,22:33])

        x_all = torch.cat([x_pd, x_di, x_fd, x_sa, x_es], dim=2)
        batch, sequence, f = x_all.shape

        x_all_out = []

        for i in range(0, sequence):
            x = self.liner(x_all[:, i, :].squeeze())
            x_all_out.append(x)

        x_all = torch.stack(x_all_out, dim=1)


        return x_all 
