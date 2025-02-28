import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleXANESExtractor(nn.Module):
    def __init__(self, in_channels=1, out_dim=32):
        super(MultiScaleXANESExtractor, self).__init__()
        self.branch1_conv1 = nn.Conv1d(in_channels, 64, kernel_size=3, padding=1)
        self.branch1_conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        
        self.branch2_conv1 = nn.Conv1d(in_channels, 64, kernel_size=5, padding=2)
        self.branch2_conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        
        self.branch3_conv1 = nn.Conv1d(in_channels, 64, kernel_size=7, padding=3)
        self.branch3_conv2 = nn.Conv1d(64, 128, kernel_size=7, padding=3)

        self.branch1_maskconv1 = nn.Conv1d(in_channels, 1, kernel_size=3, padding=1, bias=False)
        self.branch1_maskconv2 = nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=False)
        
        self.branch2_maskconv1 = nn.Conv1d(in_channels, 1, kernel_size=5, padding=2, bias=False)
        self.branch2_maskconv2 = nn.Conv1d(1, 1, kernel_size=5, padding=2, bias=False)
        
        self.branch3_maskconv1 = nn.Conv1d(in_channels, 1, kernel_size=7, padding=3, bias=False)
        self.branch3_maskconv2 = nn.Conv1d(1, 1, kernel_size=7, padding=3, bias=False)
        
        nn.init.constant_(self.branch1_maskconv1.weight, 1.0)
        nn.init.constant_(self.branch1_maskconv2.weight, 1.0)
        nn.init.constant_(self.branch2_maskconv1.weight, 1.0)
        nn.init.constant_(self.branch2_maskconv2.weight, 1.0)
        nn.init.constant_(self.branch3_maskconv1.weight, 1.0)
        nn.init.constant_(self.branch3_maskconv2.weight, 1.0)
        
        self.adaptive_pool = nn.AdaptiveAvgPool1d(10)
        self.fc = nn.Linear(128*3*10, out_dim)
        
    def forward(self, x, x_mask):
        B, N, L = x.shape
        x = x.view(B*N, 1, L)
        
        if x_mask.dim() == 2:
            x_mask = x_mask.unsqueeze(-1).float()  # (B, N, 1)
            x_mask = x_mask.expand(-1, -1, L)      # (B, N, L)
        mask_3d = x_mask.view(B*N, 1, L).float()
        
        out1 = F.relu(self.branch1_conv1(x))
        with torch.no_grad():
            m1 = self.branch1_maskconv1(mask_3d)
            m1 = (m1 >= 1).float()
        out1 = out1 * m1

        out1 = F.relu(self.branch1_conv2(out1))
        with torch.no_grad():
            m1 = self.branch1_maskconv2(m1)
            m1 = (m1 >= 1).float()
        out1 = out1 * m1

        out2 = F.relu(self.branch2_conv1(x))
        with torch.no_grad():
            m2 = self.branch2_maskconv1(mask_3d)
            m2 = (m2 >= 1).float()
        out2 = out2 * m2

        out2 = F.relu(self.branch2_conv2(out2))
        with torch.no_grad():
            m2 = self.branch2_maskconv2(m2)
            m2 = (m2 >= 1).float()
        out2 = out2 * m2

        out3 = F.relu(self.branch3_conv1(x))
        with torch.no_grad():
            m3 = self.branch3_maskconv1(mask_3d)
            m3 = (m3 >= 1).float()
        out3 = out3 * m3

        out3 = F.relu(self.branch3_conv2(out3))
        with torch.no_grad():
            m3 = self.branch3_maskconv2(m3)
            m3 = (m3 >= 1).float()
        out3 = out3 * m3
        
        cat_out = torch.cat([out1, out2, out3], dim=1)
        pooled = self.adaptive_pool(cat_out)
        pooled = pooled.view(B*N, -1)
        feat = self.fc(pooled)
        
        feat = feat.view(B, N, -1)
        
        sum_mask = mask_3d.sum(dim=(1,2))
        valid_mask_1d = (sum_mask > 0).float()
        valid_mask_2d = valid_mask_1d.view(B, N, 1)
        feat = feat * valid_mask_2d
        return feat


def map_atomic_number(atomic_number):
    if 1 <= atomic_number <= 94:
        return atomic_number
    else:
        raise ValueError(f"Atomic number {atomic_number} is not in TM range.")


class ElementsFeatureProcessor(nn.Module):
    def __init__(self):
        super(ElementsFeatureProcessor, self).__init__()
        self.float_linear = nn.Linear(5, 16)
        self.atom_embedding = nn.Embedding(95, 8)
        self.type_embedding = nn.Embedding(6, 4)

    def forward(self, elements_info, elements_mask):
        masked_info = elements_info * elements_mask.unsqueeze(-1).float()
        float_features = masked_info[:, :, :5]  # (B, N, 5)
        atomic_num = masked_info[:, :, 5].long()
        element_type = masked_info[:, :, 6].long()
        
        float_feat = F.relu(self.float_linear(float_features))  # => (B, N, 16)
        
        B, N = atomic_num.shape
        out = float_feat.new_zeros(B, N, 28)
        
        for i in range(B):
            for j in range(N):
                if elements_mask[i, j] < 0.5:
                    continue
                an = atomic_num[i, j].item()
                et = element_type[i, j].item()
                out[i, j, :16] = float_feat[i, j]
                
                if (1 <= an <= 94):
                    mapped_an = map_atomic_number(an)
                    emb_z = self.atom_embedding(
                        torch.tensor(mapped_an, device=atomic_num.device).unsqueeze(0)
                    )[0]   # shape(8,)
                    emb_t = self.type_embedding(
                        torch.tensor(et, device=atomic_num.device).unsqueeze(0)
                    )[0]   # shape(4,)
                    
                    out[i, j, 16:24] = emb_z
                    out[i, j, 24:] = emb_t
        
        out = out * elements_mask.unsqueeze(-1)
        return out


class FuseElementAndXANES(nn.Module):
    def __init__(self, xanes_dim=32, elem_dim=28, out_dim=64):
        super(FuseElementAndXANES, self).__init__()
        self.fc = nn.Linear(xanes_dim + elem_dim, out_dim)

    def forward(self, xanes_feat, elem_feat, elem_mask):
        fused = torch.cat([xanes_feat, elem_feat], dim=-1) # (B, N, 60)
        fused = self.fc(fused)                             # (B, N, 64)
        fused = fused * elem_mask.unsqueeze(-1)
        return fused


class AttentionAggregator(nn.Module):
    def __init__(self, embed_dim=64, n_heads=4):
        super(AttentionAggregator, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True)
        
    def forward(self, elem_features, elem_mask, return_attn_weights=False):
        """
        elem_features: (B, N, 64)
        elem_mask: (B, N)
        """
        key_padding_mask = ~(elem_mask.bool())
        attn_out, attn_weights = self.attn(elem_features, elem_features, elem_features,
                                           key_padding_mask=key_padding_mask)

        attn_out = attn_out * elem_mask.unsqueeze(-1).float()
        
        if return_attn_weights:
            return attn_out, attn_weights
        else:
            return attn_out


class FinalRegressor(nn.Module):
    def __init__(self, in_dim=64, hidden_dim=64, out_dim=1):
        super(FinalRegressor, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class MyNet(nn.Module):
    def __init__(self, num_classes):
        super(MyNet, self).__init__()
        self.xanes_extractor = MultiScaleXANESExtractor(in_channels=1, out_dim=32)
        self.elem_proc = ElementsFeatureProcessor()
        
        self.fuse = FuseElementAndXANES(xanes_dim=32, elem_dim=28, out_dim=64)
        
        self.attention = AttentionAggregator(embed_dim=64, n_heads=4)
        self.regressor = FinalRegressor(in_dim=64, hidden_dim=64, out_dim=num_classes)
        
    def forward(self, all_xanes, all_xanes_mask, all_elem_info, all_elem_mask, return_attn_weights=False):
        xanes_feat = self.xanes_extractor(all_xanes, all_xanes_mask)  # => (B, N, 32)
        
        elem_feat = self.elem_proc(all_elem_info, all_elem_mask)      # => (B, N, 28)
        
        fused_feat = self.fuse(xanes_feat, elem_feat, all_elem_mask)  # => (B, N, 64)
        
        if return_attn_weights:
            attn_out, attn_weights = self.attention(fused_feat, all_elem_mask, return_attn_weights=True)
        else:
            attn_out = self.attention(fused_feat, all_elem_mask, return_attn_weights=False)
            attn_weights = None
        
        sum_feat = torch.sum(attn_out, dim=1)    
        count = torch.sum(all_elem_mask, dim=1, keepdim=True) + 1e-8
        pooled = sum_feat / count                # (B, 64)
        
        out = self.regressor(pooled)
        
        if return_attn_weights:
            return out, attn_weights
        else:
            return out
