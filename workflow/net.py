import torch
import torch.nn as nn
import torch.nn.functional as F


'''Net'''
def map_TM_atomic_number(atomic_number):
    if 21 <= atomic_number <= 30:
        return atomic_number - 21 + 1
    elif 39 <= atomic_number <= 48:
        return atomic_number - 39 + 11
    else:
        raise ValueError(f"Atomic number {atomic_number} is not in TM range.")

def map_NotTM_atomic_number(atomic_number):
    if 1 <= atomic_number <= 20:
        return atomic_number
    elif 31 <= atomic_number <= 38:
        return atomic_number - 31 + 21
    elif 49 <= atomic_number <= 94:
        return atomic_number - 49 + 29
    else:
        raise ValueError(f"Atomic number {atomic_number} is not in TM range.")


class MultiScaleXANESExtractor(nn.Module):
    """
    多尺度卷积 + Mask 处理 + AdaptiveAvgPool => 最终得到 (B, TMLengthMax, out_dim)
    替代原先的 XANESExtractor
    """
    def __init__(self, in_channels=1, out_dim=32):
        super(MultiScaleXANESExtractor, self).__init__()
        
        # ----- 多分支卷积 ----- #
        # 分支1: kernel=3
        self.branch1_conv1 = nn.Conv1d(in_channels, 64, kernel_size=3, padding=1)
        self.branch1_conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        
        # 分支2: kernel=5
        self.branch2_conv1 = nn.Conv1d(in_channels, 64, kernel_size=5, padding=2)
        self.branch2_conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        
        # 分支3: kernel=7
        self.branch3_conv1 = nn.Conv1d(in_channels, 64, kernel_size=7, padding=3)
        self.branch3_conv2 = nn.Conv1d(64, 128, kernel_size=7, padding=3)

        # 对应的“mask卷积”，全1卷积，用于统计在感受野内是否全是0
        # 分支1
        self.branch1_maskconv1 = nn.Conv1d(in_channels, 1, kernel_size=3, padding=1, bias=False)
        self.branch1_maskconv2 = nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=False)
        
        # 分支2
        self.branch2_maskconv1 = nn.Conv1d(in_channels, 1, kernel_size=5, padding=2, bias=False)
        self.branch2_maskconv2 = nn.Conv1d(1, 1, kernel_size=5, padding=2, bias=False)
        
        # 分支3
        self.branch3_maskconv1 = nn.Conv1d(in_channels, 1, kernel_size=7, padding=3, bias=False)
        self.branch3_maskconv2 = nn.Conv1d(1, 1, kernel_size=7, padding=3, bias=False)
        
        # 将mask卷积权重设为全1
        nn.init.constant_(self.branch1_maskconv1.weight, 1.0)
        nn.init.constant_(self.branch1_maskconv2.weight, 1.0)
        nn.init.constant_(self.branch2_maskconv1.weight, 1.0)
        nn.init.constant_(self.branch2_maskconv2.weight, 1.0)
        nn.init.constant_(self.branch3_maskconv1.weight, 1.0)
        nn.init.constant_(self.branch3_maskconv2.weight, 1.0)
        
        # ----- 池化 + 全连接 ----- #
        # 这里我们先做AdaptiveAvgPool1d(10)，然后fc
        # 每个分支到最后会输出 128 通道 => 3分支 => cat => 128*3=384 通道
        # 池化后长度=10 => 最终特征 384*10=3840
        self.adaptive_pool = nn.AdaptiveAvgPool1d(10)
        self.fc = nn.Linear(128*3*10, out_dim)
        
    def forward(self, x, x_mask):
        """
        x:      (B, TMLengthMax, 200)
        x_mask: (B, TMLengthMax, 200) True/False (True=有效)
        
        返回: (B, TMLengthMax, out_dim)
        """

        B, N, L = x.shape  # e.g. B=batch_size, N=TMLengthMax, L=200
        
        # ========== 1) flatten到 (B*N, 1, L) ==========
        x = x.view(B*N, 1, L)                  # => (B*N, 1, 200)
        if x_mask.dim() == 2:
            x_mask = x_mask.unsqueeze(-1).float()
            x_mask = x_mask.expand(-1, -1, L)  # => (B*N, 1, 200)
        mask_3d = x_mask.view(B*N, 1, L).float()  # => (B*N, 1, 200), 1=有效,0=无效
        
        # ========== 分支1 ==========
        out1 = F.relu(self.branch1_conv1(x))
        # mask conv1
        with torch.no_grad():
            m1 = self.branch1_maskconv1(mask_3d)
            m1 = (m1 >= 1).float()
        out1 = out1 * m1  # 屏蔽无效

        out1 = F.relu(self.branch1_conv2(out1))
        with torch.no_grad():
            m1 = self.branch1_maskconv2(m1)
            m1 = (m1 >= 1).float()
        out1 = out1 * m1  # 再屏蔽

        # ========== 分支2 ==========
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

        # ========== 分支3 ==========
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
        
        # ========== 拼接 ==========
        # out1, out2, out3 => (B*N, 128, L)
        cat_out = torch.cat([out1, out2, out3], dim=1)  # => (B*N, 128*3, L=200)
        
        # ========== 池化 + fc ==========
        pooled = self.adaptive_pool(cat_out)           # => (B*N, 128*3, 10)
        # 做一个 per-channel avg => flatten => fc
        # 先 reshape
        pooled = pooled.view(B*N, -1)                  # => (B*N, 128*3*10=3840)
        feat = self.fc(pooled)                         # => (B*N, out_dim=32)
        
        # ========== 2) reshape回 (B, N, out_dim) ==========
        feat = feat.view(B, N, -1)  # => (B, TMLengthMax, out_dim)
        
        # ========== 3) 对“整条光谱全零”的无效条目再屏蔽 ==========
        #   如果一条光谱是全0, mask_3d的和=0 => 这是(1,L)都=0 => sum=0
        #   => (B*N,) 维度 => reshape => (B, N)
        #   => broadcast mul => (B, N, out_dim)
        sum_mask = mask_3d.sum(dim=(1,2))  # => (B*N)
        # 大于0 => 该条光谱有效
        valid_mask_1d = (sum_mask > 0).float()  # => (B*N,)
        valid_mask_2d = valid_mask_1d.view(B, N, 1) # => (B, N, 1)
        
        feat = feat * valid_mask_2d  # => final (B, N, out_dim)

        return feat



class ElementsFeatureProcessor(nn.Module):
    def __init__(self):
        super(ElementsFeatureProcessor, self).__init__()
        self.float_linear = nn.Linear(5, 16)
        self.TM_embedding1 = nn.Embedding(21, 8)
        self.NotTM_embedding1 = nn.Embedding(75, 8)
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
                mapped_an.append(map_TM_atomic_number(row.item()) if row>0 else 0)
            mapped_an = torch.tensor(mapped_an, device=atomic_num.device)
            mapped_an = mapped_an.view(atomic_num.shape)
            emb_z = self.TM_embedding1(mapped_an)  # (b, n, 8)
            out = torch.cat([float_feat, emb_z], dim=-1)  # (b, n, 16+8=24)
            
        else:
            mapped_an = []
            for row in atomic_num.view(-1):
                mapped_an.append(map_NotTM_atomic_number(row.item()) if row>0 else 0)
            mapped_an = torch.tensor(mapped_an, device=atomic_num.device)
            mapped_an = mapped_an.view(atomic_num.shape)
            emb_z = self.NotTM_embedding1(mapped_an)  # (b, n, 8)
            emb_t = self.NotTM_embedding2(element_type)  # (b, n, 4)
            out = torch.cat([float_feat, emb_z, emb_t], dim=-1)  # (b, n, 16+8+4=28)
        
        out = out * elements_mask.unsqueeze(-1)
        return out

class FuseTMandXANES(nn.Module):
    def __init__(self, xanes_dim=32, tm_info_dim=24, out_dim=64):
        super(FuseTMandXANES, self).__init__()
        # 把 xanes+tm_info => 64
        self.fc = nn.Linear(xanes_dim + tm_info_dim, out_dim)

    def forward(self, xanes_feat, tm_info_feat, tm_mask):
        """
        xanes_feat: (b, num_TM, xanes_dim)
        tm_info_feat: (b, num_TM, tm_info_dim)
        tm_mask: (b, num_TM)
        """
        fused = torch.cat([xanes_feat, tm_info_feat], dim=-1) # => (b, num_TM, xanes_dim+tm_info_dim)
        fused = self.fc(fused)
        # 乘 mask, 避免padding干扰
        fused = fused * tm_mask.unsqueeze(-1)
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
        self.tm_fuser = FuseTMandXANES(xanes_dim=32, tm_info_dim=24, out_dim=64)
        self.attention = AttentionAggregator(embed_dim=64, n_heads=4)
        self.regressor = FinalRegressor(in_dim=64, hidden_dim=64, out_dim=num_classes)
        
    def forward(self, TM_spec, TM_spec_mask, TM_info, TM_mask, NotTM_info, NotTM_mask, return_attn_weights=False):
        TM_xanes_feat = self.xanes_extractor(TM_spec, TM_spec_mask)
        TM_elem_feat = self.elem_proc(TM_info, TM_mask, is_tm=True)
        TM_fused_feat = self.tm_fuser(TM_xanes_feat, TM_elem_feat, TM_mask)
        NotTM_elem_feat = self.elem_proc(NotTM_info, NotTM_mask, is_tm=False)
        
        if NotTM_elem_feat.shape[-1] != 64:
            NotTM_elem_feat = F.pad(NotTM_elem_feat, (0, 64 - NotTM_elem_feat.shape[-1], 0, 0, 0, 0))
        
        all_elem_feat = torch.cat([TM_fused_feat, NotTM_elem_feat], dim=1)
        all_mask = torch.cat([TM_mask, NotTM_mask], dim=1)
        
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