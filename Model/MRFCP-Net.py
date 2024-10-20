import torch
import torch.nn as nn
import torch.nn.functional as F
from Transencoders_EEG_patch import TransformerEncoders1 as EEGencoders1
from Transencoders_EEG_patch import TransformerEncoders2 as EEGencoders2
from Transencoders_EYE_patch import TransformerEncoders1 as EYEencoders1
from Transencoders_EYE_patch import TransformerEncoders2 as EYEencoders2
from Attention_EEG_patch import Att_Layer as Att_Layer_EEG
from Attention_EEG_patch import MLP as MLP_EEG
from Attention_EYE_patch import Att_Layer as Att_Layer_EYE
from Attention_EYE_patch import MLP as MLP_EYE
from einops import rearrange


class concoder(nn.Module):
    def __init__(self):
        super(concoder, self).__init__()

        self.EEGencoder_layer1 = EEGencoders1()
        self.EYEencoder_layer1 = EYEencoders1()
        self.Brain_region_Att_layer = Att_Layer_EEG(input=64, output=64)
        self.EYE_feature_Att_layer = Att_Layer_EYE(input=64, output=64)
        self.MLP_EEG = MLP_EEG()
        self.MLP_EYE = MLP_EYE()

        self.eeg_cp_attn1 = nn.MultiheadAttention(embed_dim=10, num_heads=5, batch_first=True)
        self.eye_cp_attn1 = nn.MultiheadAttention(embed_dim=10, num_heads=5, batch_first=True)
        self.fusion_eeg_attn = nn.MultiheadAttention(embed_dim=10, num_heads=5, batch_first=True)
        self.fusion_eye_attn = nn.MultiheadAttention(embed_dim=10, num_heads=5, batch_first=True)

        self.EEGencoder_layer2 = EEGencoders2()
        self.EYEencoder_layer2 = EYEencoders2()

        self.Att_Layer_model = Att_Layer_model(10, 10)

        self.batchnorm_layer = nn.BatchNorm1d(10)

        self.fc1 = nn.Sequential(
            nn.Linear(in_features=40, out_features=80),
            nn.Linear(in_features=80, out_features=20),
        )

        self.fc2 = nn.Sequential(
            nn.Linear(in_features=20, out_features=1),
            nn.Sigmoid()
        )

        self.fc3 = nn.Sequential(
            nn.Linear(in_features=20, out_features=40),

        )
        
        self.fc4 = nn.Sequential(
            nn.Linear(in_features=40, out_features=20),
            nn.Linear(in_features=20, out_features=1),
            nn.Sigmoid()
        )



    def forward(self, x_pf, x_f, x_lt, x_c, x_rt, x_lp, x_p, x_rp, x_o, x_eye):
        x_eye[torch.isnan(x_eye)] = 0

        eeg_all_out = []
        all_br_weight = []
        eye_all_out = []
        all_ef_weight = []
        eeg_out1 = []
        eye_out1 = []

        eeg_all1 = self.EEGencoder_layer1(x_pf, x_f, x_lt, x_c, x_rt, x_lp, x_p, x_rp, x_o)
        batch, sequence, lobe, f = eeg_all1.shape
        for i in range(0, sequence):
            eeg, br_weight = self.Brain_region_Att_layer(eeg_all1[:, i, :, :].squeeze())
            eeg_all_out.append(eeg)
            all_br_weight.append(br_weight)

        eeg_all1 = torch.stack(eeg_all_out, dim=1)
        eeg_all1 = rearrange(eeg_all1, 'b s l f -> b s (l f)')

        for i in range(0, sequence):
            eeg = self.MLP_EEG(eeg_all1[:, i, :].squeeze())
            eeg_out1.append(eeg)

        eeg_out1 = torch.stack(eeg_out1, dim=1)  

        eye_all1 = self.EYEencoder_layer1(x_eye)
        for i in range(0, sequence):
            eye, ef_weight = self.EYE_feature_Att_layer(eye_all1[:, i, :, :].squeeze())
            eye_all_out.append(eye)
            all_ef_weight.append(ef_weight)

        eye_all1 = torch.stack(eye_all_out, dim=1)
        eye_all1 = rearrange(eye_all1, 'b s l f -> b s (l f)')

        for i in range(0, sequence):
            eye = self.MLP_EYE(eye_all1[:, i, :].squeeze())
            eye_out1.append(eye)

        eye_out1 = torch.stack(eye_out1, dim=1)


        device = torch.device('cuda:0')
        local_model1 = LocalModel1().to(device)
        local_model2 = LocalModel2().to(device)
        global_model = GlobalModel().to(device)

        eeg_all2 = self.EEGencoder_layer2(x_pf, x_f, x_lt, x_c, x_rt, x_lp, x_p, x_rp, x_o)

        eye_all2 = self.EYEencoder_layer2(x_eye)


        local_model1(eeg_all2)
        local_model2(eye_all2)
        params1 = local_model1.state_dict()
        params2 = local_model2.state_dict()

        average_params = {}
        for key in params1.keys():
            average_params[key] = (params1[key] + params2[key]) / 2
            
        global_model.load_state_dict(average_params)

        eeg_out2 = global_model(eeg_all2)[0]
        eye_out2 = global_model(eye_all2)[0]


        batch, sequence, f = eeg_out2.shape

        P_out,_ = self.eeg_cp_attn1(eeg_out1,eye_out1,eye_out1)
        C_out,_ = self.eye_cp_attn1(eeg_out2,eye_out2,eye_out2)

        fusion_p_atten,_ = self.fusion_eeg_attn(P_out, C_out, C_out)
        fusion_c_atten,_ = self.fusion_eye_attn(C_out, P_out, P_out)


        P = fusion_p_atten + P_out
        C = fusion_c_atten + C_out

        fusion = torch.cat((P, C), dim=2)  # b*s*20
        output = []

        for i in range(0, sequence):
            x = self.fc3(fusion[:, i, :].squeeze())
            x = self.fc4(x)
            output.append(x)

        output = torch.hstack(output)


        return output


class Att_Layer_model(nn.Module):

    def __init__(self, input, output):
        super(Att_Layer_model, self).__init__()

        self.P_linear = nn.Linear(input, output, bias=True)
        self.V_linear = nn.Linear(input, 3, bias=False)

    def forward(self, att_input):
        P = self.P_linear(att_input)

        feature = torch.tanh(P)
        alpha = self.V_linear(feature)
        alpha = F.softmax(alpha, dim=2)
        out = torch.matmul(alpha, att_input)
        return out, alpha


class LocalModel1(nn.Module):
    def __init__(self):
        super(LocalModel1, self).__init__()
        self.biLSTM = nn.LSTM(
            bidirectional=False,
            batch_first=True,
            input_size=10,
            hidden_size=10,
            num_layers=5
        )


    def forward(self, x):

        return self.biLSTM(x)

class LocalModel2(nn.Module):
    def __init__(self):
        super(LocalModel2, self).__init__()
        self.biLSTM = nn.LSTM(
            bidirectional=False,
            batch_first=True,
            input_size=10,
            hidden_size=10,
            num_layers=5
        )


    def forward(self, x):

        return self.biLSTM(x)


class GlobalModel(nn.Module):
    def __init__(self):
        super(GlobalModel, self).__init__()
        self.biLSTM = nn.LSTM(
            bidirectional=False,
            batch_first=True,
            input_size=10,
            hidden_size=10,
            num_layers=5
        )


    def forward(self, x):

        return self.biLSTM(x)
