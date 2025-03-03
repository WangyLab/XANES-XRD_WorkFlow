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
            x_mask = x_mask.unsqueeze(-1).float()
            x_mask = x_mask.expand(-1, -1, L)
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
    
RARE_MIN = 57  # La
RARE_MAX = 80  # Hg

def map_Rare_atomic_number(Z: int) -> int:
    if RARE_MIN <= Z <= RARE_MAX:
        return Z - RARE_MIN + 1  # => 1..24
    else:
        return 0


def map_NotRare_atomic_number(Z: int) -> int:
    if 1 <= Z <= 56:
        return Z
    elif 57 <= Z <= 80:
        return 0
    elif 81 <= Z <= 118:
        return (Z - 80) + 56
    else:
        return 0

class ElementsFeatureProcessor(nn.Module):
    def __init__(self):
        super(ElementsFeatureProcessor, self).__init__()
        self.float_linear = nn.Linear(5, 16)
        self.TM_embedding1 = nn.Embedding(25, 8)
        self.NotTM_embedding1 = nn.Embedding(95, 8)
        self.NotTM_embedding2 = nn.Embedding(6, 4)

    def forward(self, elements_info, elements_mask, is_tm=True):
        masked_info = elements_info * elements_mask.unsqueeze(-1).float()
        float_features = masked_info[:, :, :5]
        atomic_num = masked_info[:, :, 5].long()
        element_type = masked_info[:, :, 6].long()
        
        float_feat = F.relu(self.float_linear(float_features))
        if is_tm:
            mapped_an = []
            for row in atomic_num.view(-1):
                mapped_an.append(map_Rare_atomic_number(row.item()) if row>0 else 0)
            mapped_an = torch.tensor(mapped_an, device=atomic_num.device)
            mapped_an = mapped_an.view(atomic_num.shape)
            emb_z = self.TM_embedding1(mapped_an)
            out = torch.cat([float_feat, emb_z], dim=-1)  
        else:
            mapped_an = []
            for row in atomic_num.view(-1):
                mapped_an.append(map_NotRare_atomic_number(row.item()) if row>0 else 0)
            mapped_an = torch.tensor(mapped_an, device=atomic_num.device)
            mapped_an = mapped_an.view(atomic_num.shape)
            emb_z = self.NotTM_embedding1(mapped_an)
            emb_t = self.NotTM_embedding2(element_type)
            out = torch.cat([float_feat, emb_z, emb_t], dim=-1)
        
        out = out * elements_mask.unsqueeze(-1)
        return out

class FuseTMandXANES(nn.Module):
    def __init__(self, xanes_dim=32, em_info_dim=24, out_dim=64):
        super(FuseTMandXANES, self).__init__()
        self.fc = nn.Linear(xanes_dim + em_info_dim, out_dim)

    def forward(self, xanes_feat, em_info_feat, em_mask):
        """
        xanes_feat: (b, num_TM, xanes_dim)
        tm_info_feat: (b, num_TM, tm_info_dim)
        tm_mask: (b, num_TM)
        """
        fused = torch.cat([xanes_feat, em_info_feat], dim=-1) # => (b, num_TM, xanes_dim+tm_info_dim)
        fused = self.fc(fused)
        fused = fused * em_mask.unsqueeze(-1)
        return fused

class AttentionAggregator(nn.Module):
    def __init__(self, embed_dim=64, n_heads=4):
        super(AttentionAggregator, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True)
        
    def forward(self, elem_features, elem_mask, return_attn_weights=False):
        key_padding_mask = ~(elem_mask.bool())
        attn_out, attn_weights = self.attn(elem_features, elem_features, elem_features, key_padding_mask=key_padding_mask)
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
        self.em_fuser = FuseTMandXANES(xanes_dim=32, em_info_dim=24, out_dim=64)
        self.attention = AttentionAggregator(embed_dim=64, n_heads=4)
        self.regressor = FinalRegressor(in_dim=64, hidden_dim=64, out_dim=num_classes)
        
    def forward(self, EM_spec, EM_spec_mask, EM_info, EM_mask, NotEM_info, NotEM_mask, return_attn_weights=False):
        EM_xanes_feat = self.xanes_extractor(EM_spec, EM_spec_mask)
        EM_elem_feat = self.elem_proc(EM_info, EM_mask, is_em=True)
        EM_fused_feat = self.tm_fuser(EM_xanes_feat, EM_elem_feat, EM_mask)
        NotEM_elem_feat = self.elem_proc(NotEM_info, NotEM_mask, is_em=False)
        
        if NotEM_elem_feat.shape[-1] != 64:
            NotEM_elem_feat = F.pad(NotEM_elem_feat, (0, 64 - NotEM_elem_feat.shape[-1], 0, 0, 0, 0))
        
        all_elem_feat = torch.cat([EM_fused_feat, NotEM_elem_feat], dim=1)
        all_mask = torch.cat([EM_mask, NotEM_mask], dim=1)
        
        if return_attn_weights:
            attn_out, attn_weights = self.attention(all_elem_feat, all_mask, return_attn_weights=True)
        else:
            attn_out = self.attention(all_elem_feat, all_mask, return_attn_weights=False)
            attn_weights = None
        
        sum_feat = torch.sum(attn_out, dim=1)    # (b, 64)
        count = torch.sum(all_mask, dim=1, keepdim=True) + 1e-8
        pooled = sum_feat / count                # (b, 64)
        
        out = self.regressor(pooled)
        
        if return_attn_weights:
            return out, attn_weights
        else:
            return out