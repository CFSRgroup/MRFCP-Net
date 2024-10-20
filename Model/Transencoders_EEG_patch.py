import torch
import torch.nn as nn

class TransformerEncoders1(nn.Module):
    def __init__(self):
        super(TransformerEncoders1, self).__init__()
        N = [0, 5, 9, 6, 10, 6, 6, 9, 6, 5]
        self.N = N

        self.encoder_pf = nn.TransformerEncoderLayer(
            d_model=N[1]*5,
            nhead=5,
            dim_feedforward=64,
            batch_first=True
        )
        self.liner_pf = nn.Linear(
            in_features= N[1]*5,
            out_features= 64

        )

        self.encoder_f = nn.TransformerEncoderLayer(
            d_model=N[2] * 5,
            nhead=5,
            dim_feedforward=64,
            batch_first=True
        )
        self.liner_f = nn.Linear(
            in_features=N[2] * 5,
            out_features= 64

        )

        self.encoder_lt = nn.TransformerEncoderLayer(
            d_model=N[3] * 5,
            nhead=5,
            dim_feedforward=64,
            batch_first=True
        )
        self.liner_lt = nn.Linear(
            in_features=N[3] * 5,
            out_features= 64

        )

        self.encoder_c = nn.TransformerEncoderLayer(
            d_model=N[4] * 5,
            nhead=5,
            dim_feedforward=64,
            batch_first=True
        )
        self.liner_c = nn.Linear(
            in_features=N[4] * 5,
            out_features= 64

        )

        self.encoder_rt = nn.TransformerEncoderLayer(
            d_model=N[5] * 5,
            nhead=5,
            dim_feedforward=64,
            batch_first=True
        )
        self.liner_rt = nn.Linear(
            in_features=N[5] * 5,
            out_features= 64

        )

        self.encoder_lp = nn.TransformerEncoderLayer(
            d_model=N[6] * 5,
            nhead=5,
            dim_feedforward=64,
            batch_first=True
        )
        self.liner_lp = nn.Linear(
            in_features=N[6] * 5,
            out_features= 64

        )

        self.encoder_p = nn.TransformerEncoderLayer(
            d_model=N[7] * 5,
            nhead=5,
            dim_feedforward=64,
            batch_first=True
        )
        self.liner_p = nn.Linear(
            in_features=N[7] * 5,
            out_features= 64

        )

        self.encoder_rp = nn.TransformerEncoderLayer(
            d_model=N[8] * 5,
            nhead=5,
            dim_feedforward=64,
            batch_first=True
        )
        self.liner_rp = nn.Linear(
            in_features=N[8] * 5,
            out_features= 64

        )

        self.encoder_o = nn.TransformerEncoderLayer(
            d_model=N[9] * 5,
            nhead=5,
            dim_feedforward=64,
            batch_first=True
        )
        self.liner_o = nn.Linear(
            in_features=N[9] * 5,
            out_features= 64

        )



    def forward(self, x_pf, x_f, x_lt, x_c, x_rt, x_lp, x_p, x_rp, x_o):
        batch, sequence, _, _ = x_pf.shape

        x_pf = torch.reshape(x_pf, (batch, sequence, self.N[1]*5))
        x_pf = self.encoder_pf(x_pf)
        x_f = torch.reshape(x_f, (batch, sequence, self.N[2] * 5))
        x_f = self.encoder_f(x_f)
        x_lt = torch.reshape(x_lt, (batch, sequence, self.N[3] * 5))
        x_lt = self.encoder_lt(x_lt)
        x_c = torch.reshape(x_c, (batch, sequence, self.N[4] * 5))
        x_c = self.encoder_c(x_c)
        x_rt = torch.reshape(x_rt, (batch, sequence, self.N[5] * 5))
        x_rt = self.encoder_rt(x_rt)
        x_lp = torch.reshape(x_lp, (batch, sequence, self.N[6] * 5))
        x_lp = self.encoder_lp(x_lp)
        x_p = torch.reshape(x_p, (batch, sequence, self.N[7] * 5))
        x_p = self.encoder_p(x_p)
        x_rp = torch.reshape(x_rp, (batch, sequence, self.N[8] * 5))
        x_rp = self.encoder_rp(x_rp)
        x_o = torch.reshape(x_o, (batch, sequence, self.N[9] * 5))
        x_o = self.encoder_o(x_o)

        x_pf_all = []
        x_f_all = []
        x_lt_all = []
        x_c_all = []
        x_rt_all = []
        x_lp_all = []
        x_p_all = []
        x_rp_all = []
        x_o_all = []

        for i in range(0, sequence):


            x_pf_out = self.liner_pf(x_pf[:, i, :].squeeze())
            x_f_out = self.liner_f(x_f[:, i, :].squeeze())
            x_lt_out = self.liner_lt(x_lt[:, i, :].squeeze())
            x_c_out = self.liner_c(x_c[:, i, :].squeeze())
            x_rt_out = self.liner_rt(x_rt[:, i, :].squeeze())
            x_lp_out = self.liner_lp(x_lp[:, i, :].squeeze())
            x_p_out = self.liner_p(x_p[:, i, :].squeeze())
            x_rp_out = self.liner_rp(x_rp[:, i, :].squeeze())
            x_o_out = self.liner_o(x_o[:, i, :].squeeze())

            x_pf_all.append(x_pf_out)
            x_f_all.append(x_f_out)
            x_lt_all.append(x_lt_out)
            x_c_all.append(x_c_out)
            x_rt_all.append(x_rt_out)
            x_lp_all.append(x_lp_out)
            x_p_all.append(x_p_out)
            x_rp_all.append(x_rp_out)
            x_o_all.append(x_o_out)

        x_pf = torch.stack(x_pf_all,dim=1)
        x_f = torch.stack(x_f_all,dim=1)
        x_lt = torch.stack(x_lt_all,dim=1)
        x_c = torch.stack(x_c_all,dim=1)
        x_rt = torch.stack(x_rt_all,dim=1)
        x_lp = torch.stack(x_lp_all,dim=1)
        x_p = torch.stack(x_p_all,dim=1)
        x_rp = torch.stack(x_rp_all,dim=1)
        x_o = torch.stack(x_o_all,dim=1)

        x_all = torch.stack([x_pf, x_f, x_lt, x_c, x_rt,
                              x_lp, x_p, x_rp, x_o], dim=2)

        return x_all #b*s*9*64


class TransformerEncoders2(nn.Module):
    def __init__(self):
        super(TransformerEncoders2, self).__init__()
        N = [0, 5, 9, 6, 10, 6, 6, 9, 6, 5]
        self.N = N
        self.encoder_pf = nn.TransformerEncoderLayer(
            d_model=N[1]*5,
            nhead=5,
            dim_feedforward=64,
            batch_first=True
        )
        self.encoder_f = nn.TransformerEncoderLayer(
            d_model=N[2] * 5,
            nhead=5,
            dim_feedforward=64,
            batch_first=True
        )
        self.encoder_lt = nn.TransformerEncoderLayer(
            d_model=N[3] * 5,
            nhead=5,
            dim_feedforward=64,
            batch_first=True
        )
        self.encoder_c = nn.TransformerEncoderLayer(
            d_model=N[4] * 5,
            nhead=5,
            dim_feedforward=64,
            batch_first=True
        )
        self.encoder_rt = nn.TransformerEncoderLayer(
            d_model=N[5] * 5,
            nhead=5,
            dim_feedforward=64,
            batch_first=True
        )
        self.encoder_lp = nn.TransformerEncoderLayer(
            d_model=N[6] * 5,
            nhead=5,
            dim_feedforward=64,
            batch_first=True
        )
        self.encoder_p = nn.TransformerEncoderLayer(
            d_model=N[7] * 5,
            nhead=5,
            dim_feedforward=64,
            batch_first=True
        )
        self.encoder_rp = nn.TransformerEncoderLayer(
            d_model=N[8] * 5,
            nhead=5,
            dim_feedforward=64,
            batch_first=True
        )
        self.encoder_o = nn.TransformerEncoderLayer(
            d_model=N[9] * 5,
            nhead=5,
            dim_feedforward=64,
            batch_first=True
        )

        self.liner = nn.Sequential(
                   nn.Linear(in_features=310, out_features=10),
                   nn.BatchNorm1d(10)
                   )


    def forward(self, x_pf, x_f, x_lt, x_c, x_rt, x_lp, x_p, x_rp, x_o):
        batch, sequence, _, _ = x_pf.shape

        x_pf = torch.reshape(x_pf, (batch, sequence, self.N[1]*5))
        x_pf = self.encoder_pf(x_pf)
        x_f = torch.reshape(x_f, (batch, sequence, self.N[2] * 5))
        x_f = self.encoder_f(x_f)
        x_lt = torch.reshape(x_lt, (batch, sequence, self.N[3] * 5))
        x_lt = self.encoder_lt(x_lt)
        x_c = torch.reshape(x_c, (batch, sequence, self.N[4] * 5))
        x_c = self.encoder_c(x_c)
        x_rt = torch.reshape(x_rt, (batch, sequence, self.N[5] * 5))
        x_rt = self.encoder_rt(x_rt)
        x_lp = torch.reshape(x_lp, (batch, sequence, self.N[6] * 5))
        x_lp = self.encoder_lp(x_lp)
        x_p = torch.reshape(x_p, (batch, sequence, self.N[7] * 5))
        x_p = self.encoder_p(x_p)
        x_rp = torch.reshape(x_rp, (batch, sequence, self.N[8] * 5))
        x_rp = self.encoder_rp(x_rp)
        x_o = torch.reshape(x_o, (batch, sequence, self.N[9] * 5))
        x_o = self.encoder_o(x_o)


        x_all = torch.cat([x_pf, x_f, x_lt, x_c, x_rt,
                              x_lp, x_p, x_rp, x_o], dim=2)

        batch, sequence, f = x_all.shape

        x_all_out = []

        for i in range(0, sequence):
            x = self.liner(x_all[:, i, :].squeeze())
            x_all_out.append(x)

        x_all = torch.stack(x_all_out, dim=1)


        return x_all






