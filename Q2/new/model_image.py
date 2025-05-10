import torch
import torch.nn as nn
from torchsummary import summary
from torchviz import make_dot

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class ChromaticAttention(nn.Module):
    """光谱注意力机制"""
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c = x.size()
        y = self.avg_pool(x.unsqueeze(-1)).view(b, c)
        y = self.fc(y).view(b, c)
        return x * y.expand_as(x)

class GamutProjection(nn.Module):
    """可微分色域投影层（最终正确版）"""
    def __init__(self):
        super().__init__()
        # 可学习的5x5投影矩阵
        self.proj_matrix = nn.Parameter(torch.eye(5))  # 初始化为单位矩阵
        
    def forward(self, x):
        # 保持5通道维度
        x = torch.mm(x, self.proj_matrix)  # [batch,5] x [5,5] → [batch,5]
        return torch.sigmoid(x)

class MLP(nn.Module):
    def __init__(self, input_dim=4, output_dim=5, hidden_dim=512):
        super().__init__()
        # 输入编码层
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        # 特征转换层
        self.transformer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2)
        )
        
        # 输出解码层
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, output_dim)
        )
        
        # 色域投影层（显示设备矩阵应为3x5）
        self.gamut_layer = GamutProjection()  # 直接传入原始矩阵
        
    def forward(self, x):
        # 输入标准化
        x = x.clamp(0, 1)
        
        # 特征编码
        x = self.encoder(x)
        
        # 特征变换
        x = self.transformer(x) + x  # 残差连接
        
        # 解码输出（5通道）
        x = self.decoder(x)
        
        # 色域投影（保持5通道）
        return self.gamut_layer(x).clamp(0, 1)

# 创建模型实例
model = MLP(input_dim=4, output_dim=5, hidden_dim=512).to(device)

# 使用torchsummary查看模型结构
summary(model, input_size=(4,))

# 使用graphviz生成可视化图形
x = torch.randn(1, 4).to(device)  # 创建一个随机输入
dot = make_dot(model(x))
dot.render('mlp_model', format='png', cleanup=True)