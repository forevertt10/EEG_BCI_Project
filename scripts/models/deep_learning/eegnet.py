import torch
from torch import nn


class EEGNet(nn.Module):

    def __init__(self, num_classes=4):

        super().__init__()


        # 第一阶段：
        # Temporal filtering
        self.firstconv = nn.Sequential(

            nn.Conv2d(
                1,
                16,
                kernel_size=(1,64),
                padding=(0,32),
                bias=False
            ),

            nn.BatchNorm2d(16)

        )


        # 第二阶段：
        # Spatial filtering
        self.depthwiseConv = nn.Sequential(

            nn.Conv2d(
                16,
                32,
                kernel_size=(22,1),
                groups=16,
                bias=False
            ),

            nn.BatchNorm2d(32),

            nn.ELU(),

            nn.AvgPool2d(
                kernel_size=(1,4)
            ),

            nn.Dropout(0.25)

        )


        # 第三阶段：
        # Separable convolution

        self.separableConv = nn.Sequential(

            nn.Conv2d(
                32,
                32,
                kernel_size=(1,16),
                padding=(0,8),
                groups=32,
                bias=False
            ),

            nn.Conv2d(
                32,
                32,
                kernel_size=(1,1),
                bias=False
            ),

            nn.BatchNorm2d(32),

            nn.ELU(),

            nn.AvgPool2d(
                kernel_size=(1,8)
            ),

            nn.Dropout(0.25)

        )


        self.classifier = nn.Linear(
            32,
            num_classes
        )


    def forward(self,x):

        x=self.firstconv(x)

        x=self.depthwiseConv(x)

        x=self.separableConv(x)

        x=torch.mean(
            x,
            dim=(2,3)
        )


        x=self.classifier(x)

        return x
if __name__ == "__main__":


    model = EEGNet()


    x=torch.randn(
        8,
        1,
        22,
        876
    )


    y=model(x)


    print("Input:")
    print(x.shape)


    print("----------------")


    print("Output:")
    print(y.shape)