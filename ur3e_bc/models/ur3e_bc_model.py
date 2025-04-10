import torch
import torch.nn as nn
from einops import rearrange
from timm.models.layers import Mlp
import timm





class UR3EBCModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Vision model
        self.vp_num = 3
        self.hist_num = 3
        self.img_chans = 3
        self.vision = timm.create_model(model_name='resnet18', pretrained=True, num_classes=0, in_chans=self.hist_num*self.img_chans)

        # Embbeded layer
        self.vision_output_dim  = 512 * self.vp_num
        self.embed_dim = 512
        self.embed_layer = nn.Linear(self.vision_output_dim, self.embed_dim)

        # Concaternate with state
        self.state_size = 13
        

        # MLP model
        self.rep_dim = self.embed_dim+ (self.state_size*self.hist_num) # output vector dimension after embeded and concatenate all wtih states
        self.output_dim  = 10
        self.act_layer = nn.ReLU
        self.mlp = Mlp(self.rep_dim, out_features=self.output_dim, act_layer=self.act_layer)

        # Output layers
        self.vel_head = nn.Linear(self.output_dim, 6)
        self.pose_head = nn.Linear(self.output_dim, 7)
        self.state_head = nn.Linear(self.output_dim, 3)

   
    def forward(self, front_im, side_im, hand_im, ee_pose, joint_state):
        # Inputs: (B, hist_num, H, W, C)
        # State: (B, hist_num, 13)

        # Stack the views into one tensor: (B, V=3, hist_num, H, W, C)
        im = torch.stack([front_im, side_im, hand_im], dim=1)

        # Rearrange into: (B, V, C * hist_num, H, W)
        # - Move channels last to first, and flatten across history
        im = rearrange(im, 'b v t h w c -> b v (t c) h w')

        # Flatten batch and view: (B * V, C * hist_num, H, W)
        x = rearrange(im, 'b v c h w -> (b v) c h w')

        # Vision encoder (e.g., resnet18)
        h = self.vision(x)  # → (B * V, 512)

        # Reshape back to (B, V * 512)
        h = rearrange(h, '(b v) d -> b (v d)', v=self.vp_num)

        # Embed to fixed dim (B, embed_dim)
        r = self.embed_layer(h)

        # merge states
        state = torch.cat([ee_pose, joint_state], dim=-1)
        # Flatten state history: (B, hist_num * state_size)
        s = rearrange(state, 'b t d -> b (t d)')

        # Concatenate vision rep and state: (B, rep_dim)
        combined = torch.cat([r, s], dim=-1)

        # MLP projection
        y = self.mlp(combined)  # (B, output_dim)

        # Output heads
        vel = self.vel_head(y)         # (B, 6)
        pose = self.pose_head(y)       # (B, 7)
        est_state = self.state_head(y) # (B, 3)

        return vel, pose, est_state


    