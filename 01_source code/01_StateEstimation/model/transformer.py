"""
created by
"""
import torch
import torch.nn as nn


class TransformerRegressor(nn.Module):
    def __init__(self, d_model, nhead, num_layers, tactile_modal, chamber_modal, pressure_modal):
        super(TransformerRegressor, self).__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.chamber_modal = chamber_modal
        self.pressure_modal = pressure_modal
        self.tactile_modal = tactile_modal

        # tactile
        if self.tactile_modal:
            self.conv_tactile_th = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1, padding=0)
            self.relu1_tactile_th = nn.ReLU()
            self.pool_tactile_th = nn.AvgPool2d(kernel_size=4, stride=4)
            self.flatten_tactile_th = nn.Flatten()
            self.conv_tactile_ff = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1, padding=0)
            self.relu1_tactile_ff = nn.ReLU()
            self.pool_tactile_ff = nn.AvgPool2d(kernel_size=4, stride=4)
            self.flatten_tactile_ff = nn.Flatten()
            self.conv_tactile_mf = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1, padding=0)
            self.relu1_tactile_mf = nn.ReLU()
            self.pool_tactile_mf = nn.AvgPool2d(kernel_size=4, stride=4)
            self.flatten_tactile_mf = nn.Flatten()

        # chamber
        if self.chamber_modal:
            self.fc_chamber_th = nn.Linear(in_features=4, out_features=1)
            self.flatten_chamber_th = nn.Flatten()
            self.fc_chamber_ff = nn.Linear(in_features=4, out_features=1)
            self.flatten_chamber_ff = nn.Flatten()
            self.fc_chamber_mf = nn.Linear(in_features=4, out_features=1)
            self.flatten_chamber_mf = nn.Flatten()

        # pressure
        if self.pressure_modal:
            self.fc_press = nn.Linear(in_features=3, out_features=int(self.d_model/4))
            self.flatten_press = nn.Flatten()

        # fusion
        if self.tactile_modal:
            self.fusion_tactile = nn.Linear(in_features=6, out_features=int(self.d_model/2))
        if self.chamber_modal:
            self.fusion_chamber = nn.Linear(in_features=3, out_features=int(self.d_model/4))

        # transformer
        self.fc1_fusion = nn.Linear(in_features=int(self.tactile_modal * self.d_model/2) +
                                                int(self.chamber_modal * self.d_model/4) +
                                                int(self.pressure_modal * self.d_model/4),
                                    out_features=self.d_model)
        self.relu_fusion = nn.ReLU()
        encoder_layers = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer=encoder_layers, num_layers=self.num_layers)

        # output
        self.fc2_fusion_th = nn.Linear(self.d_model, 3 * 5)
        self.fc2_fusion_ff = nn.Linear(self.d_model, 3 * 5)
        self.fc2_fusion_mf = nn.Linear(self.d_model, 3 * 5)


    def forward(self, input_tactile_th, input_tactile_ff, input_tactile_mf, input_chamber_th, input_chamber_ff, input_chamber_mf, input_press):
        input_tactile_th = input_tactile_th.view(-1, 1, 8, 4)  # [B, C=1, H=8, W=4]
        input_tactile_ff = input_tactile_ff.view(-1, 1, 8, 4)  # [B, C=1, H=8, W=4]
        input_tactile_mf = input_tactile_mf.view(-1, 1, 8, 4)  # [B, C=1, H=8, W=4]
        input_chamber_th = input_chamber_th.view(-1, 1, 4)  # [B, C=1, H=4]
        input_chamber_ff = input_chamber_ff.view(-1, 1, 4)  # [B, C=1, H=4]
        input_chamber_mf = input_chamber_mf.view(-1, 1, 4)  # [B, C=1, H=4]
        input_press = input_press.view(-1, 1, 3)  # [B, C=1, H=3]

        # tactile
        if self.tactile_modal:
            tactile_feature_th = self.conv_tactile_th(input_tactile_th)  # [B, C=1, H=8, W=4]
            tactile_feature_th = self.relu1_tactile_th(tactile_feature_th)
            tactile_feature_th = self.pool_tactile_th(tactile_feature_th)  # [B, C=1, H=2, W=1]
            tactile_feature_th = self.flatten_tactile_th(tactile_feature_th)  # [B, 2]
            tactile_feature_ff = self.conv_tactile_ff(input_tactile_ff)  # [B, C=1, H=8, W=4]
            tactile_feature_ff = self.relu1_tactile_ff(tactile_feature_ff)
            tactile_feature_ff = self.pool_tactile_ff(tactile_feature_ff)  # [B, C=1, H=2, W=1]
            tactile_feature_ff = self.flatten_tactile_ff(tactile_feature_ff)  # [B, 2]
            tactile_feature_mf = self.conv_tactile_mf(input_tactile_mf)  # [B, C=1, H=8, W=4]
            tactile_feature_mf = self.relu1_tactile_mf(tactile_feature_mf)
            tactile_feature_mf = self.pool_tactile_mf(tactile_feature_mf)  # [B, C=1, H=2, W=1]
            tactile_feature_mf = self.flatten_tactile_mf(tactile_feature_mf)  # [B, 2]

        # chamber
        if self.chamber_modal:
            chamber_feature_th = self.fc_chamber_th(input_chamber_th)  # [B, 1, 1]
            chamber_feature_th = self.flatten_chamber_th(chamber_feature_th)  # [B, 1]
            chamber_feature_ff = self.fc_chamber_ff(input_chamber_ff)  # [B, 1, 1]
            chamber_feature_ff = self.flatten_chamber_ff(chamber_feature_ff)  # [B, 1]
            chamber_feature_mf = self.fc_chamber_mf(input_chamber_mf)  # [B, 1, 1]
            chamber_feature_mf = self.flatten_chamber_mf(chamber_feature_mf)  # [B, 1]

        # pressure
        if self.pressure_modal:
            press_feature = self.fc_press(input_press)  # [B, 1, 1]
            press_feature = self.flatten_press(press_feature)  # [B, 1]

        # fusion
        if self.tactile_modal:
            tactile_fusion_feature = torch.cat([tactile_feature_th, tactile_feature_ff, tactile_feature_mf],
                                               dim=-1)  # [B, 6]
            tactile_fusion_feature = self.fusion_tactile(tactile_fusion_feature)  # [B, 1]
        if self.chamber_modal:
            chamber_fusion_feature = torch.cat([chamber_feature_th, chamber_feature_ff, chamber_feature_mf],
                                               dim=-1)  # [B, 3]
            chamber_fusion_feature = self.fusion_chamber(chamber_fusion_feature)  # [B, 1]
        if self.tactile_modal and self.chamber_modal and self.pressure_modal:
            concat_feature = [tactile_fusion_feature, chamber_fusion_feature, press_feature]
        elif self.tactile_modal and self.chamber_modal and not self.pressure_modal:
            concat_feature = [tactile_fusion_feature, chamber_fusion_feature]
        elif self.tactile_modal and not self.chamber_modal and self.pressure_modal:
            concat_feature = [tactile_fusion_feature, press_feature]
        elif not self.tactile_modal and self.chamber_modal and self.pressure_modal:
            concat_feature = [chamber_fusion_feature, press_feature]
        elif self.tactile_modal and not self.chamber_modal and not self.pressure_modal:
            concat_feature = [tactile_fusion_feature]
        elif not self.tactile_modal and self.chamber_modal and not self.pressure_modal:
            concat_feature = [chamber_fusion_feature]
        elif not self.tactile_modal and not self.chamber_modal and self.pressure_modal:
            concat_feature = [press_feature]
        fusion_feature = torch.cat(tensors=concat_feature, dim=-1)  # [B, 3]

        # transformer
        fusion_feature = self.fc1_fusion(fusion_feature)  # [B, 8]
        fusion_feature = self.relu_fusion(fusion_feature)
        fusion_feature = self.transformer_encoder(fusion_feature)

        # output thumb state
        output_feature_th = self.fc2_fusion_th(fusion_feature)  # [B, 15]
        output_th = output_feature_th.view(-1, 5, 3)  # [B, 5, 3]

        # output first finger state
        output_feature_ff = self.fc2_fusion_ff(fusion_feature)  # [B, 15]
        output_ff = output_feature_ff.view(-1, 5, 3)  # [B, 5, 3]

        # output middle finger state
        output_feature_mf = self.fc2_fusion_mf(fusion_feature)  # [B, 15]
        output_mf = output_feature_mf.view(-1, 5, 3)  # [B, 5, 3]

        return output_th, output_ff, output_mf
