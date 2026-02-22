import jittor as jt
import jittor.nn as nn
import utils.ResNet_for_32 as resnet_s

 
def normalize(x, dim=-1):
    """归一化函数，与F.normalize兼容"""
    return jt.normalize(x, dim=dim)


class F:
    @staticmethod
    def normalize(x, dim=-1):
        return normalize(x, dim=dim)

class SimCLR_encoder(nn.Module):
    def __init__(self, feature_dim=128):
        super(SimCLR_encoder, self).__init__()

        self.f = resnet_s.resnet50(num_input_channels=3, num_classes=2048)
        # projection head
        self.g = nn.Sequential(nn.Linear(2048, 512, bias=False), nn.BatchNorm1d(512),
                               nn.ReLU(), nn.Linear(512, feature_dim, bias=True))
        self.feature_dim = feature_dim

    def execute(self, x):
        print(f"SimCLR_encoder input type: {type(x)}")
        print(f"SimCLR_encoder input shape: {x.shape if hasattr(x, 'shape') else 'Unknown'}")

        if not isinstance(x, jt.Var):
            print("Converting input to jt.Var")
            x = jt.array(x)
            print(f"After conversion shape: {x.shape}")
 
        if len(x.shape) == 4:
            print(f"Input channels: {x.shape[1]}")
            if x.shape[1] != 3:
                print(f"Adjusting channels from {x.shape[1]} to 3")
        
                if x.shape[1] == 1:
                    # 灰度图转RGB
                    x = jt.concat([x, x, x], dim=1)
                elif x.shape[1] > 3:
                    # 取前3个通道
                    x = x[:, :3, :, :]
                print(f"After channel adjustment shape: {x.shape}")
        elif len(x.shape) != 4:
            print(f"Unexpected input shape: {x.shape}")
            # 尝试调整形状为4维
            if len(x.shape) == 3:
                # 添加批次维度
                x = x.unsqueeze(0)
                print(f"After adding batch dimension shape: {x.shape}")
        # 前向传播
        print("Calling self.f(x)")
        x = self.f(x)
        print(f"After self.f(x) shape: {x.shape}")
        feature = jt.flatten(x, start_dim=1)
        print(f"After flatten shape: {feature.shape}")
        out = self.g(feature)
        print(f"After self.g(feature) shape: {out.shape}")
        # return F.normalize(feature, dim=-1), F.normalize(out, dim=-1)
        normalized_out = F.normalize(feature, dim=-1)
        print(f"After normalization shape: {normalized_out.shape}")
        return normalized_out


    def forward(self, x):
        return self.execute(x)
