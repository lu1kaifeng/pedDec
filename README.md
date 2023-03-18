# A Pedestrian Detection and Tracking Solution

For [China Software Cup 2021 - Pedestrian Tracking Competition](https://aistudio.baidu.com/aistudio/competition/detail/73/0/introduction)

> In accordance to competition requirements, this solution is built with PaddlePaddle, a deep learning framework by Baidu.

## Implementation

> This solution utilizes RCNN for pedestrian detection, A DeepSort implementation augmented with metric learning for tracking and ReID

### Project Structure

| Source              |               Description               |  
|---------------------|:---------------------------------------:|
| deepsort_eval.py    |        DeepSort ReID evaluation         |
| deepsort_test.py    |          DeepSort ReID testing          |
| model_eval.py       |       detection model evaluation        |
| reid_pcb_train.py   |         DeepSort ReID training          |
| test.py             |    Solution Testing / Visualization     |
| mot2voc.py          |     dataset label format converter      |
| gen_gt_list.py      |     generate output for competition     |
| deep_sort/deepaddle | DeepSort implementation in PaddlePaddle |
| model_train/        |        model training in PaddleX        |
| server/             |      a web server for demostration      |

### ReID Feature Extractor Implementation

*A fully convolutional feature extractor*

```python
class BasicBlock(nn.Layer):
    def __init__(self, c_in, c_out, is_downsample=False):
        super(BasicBlock, self).__init__()
        self.is_downsample = is_downsample
        if is_downsample:
            self.conv1 = nn.Conv2D(c_in, c_out, 3, stride=2, padding=1, bias_attr=False)
        else:
            self.conv1 = nn.Conv2D(c_in, c_out, 3, stride=1, padding=1, bias_attr=False)
        self.bn1 = nn.BatchNorm2D(c_out)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2D(c_out, c_out, 3, stride=1, padding=1, bias_attr=False)
        self.bn2 = nn.BatchNorm2D(c_out)
        if is_downsample:
            self.downsample = nn.Sequential(
                nn.Conv2D(c_in, c_out, 1, stride=2, bias_attr=False),
                nn.BatchNorm2D(c_out)
            )
        elif c_in != c_out:
            self.downsample = nn.Sequential(
                nn.Conv2D(c_in, c_out, 1, stride=1, bias_attr=False),
                nn.BatchNorm2D(c_out)
            )
            self.is_downsample = True

    def forward(self, x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.relu(y)
        y = self.conv2(y)
        y = self.bn2(y)
        if self.is_downsample:
            x = self.downsample(x)
        return fluid.layers.relu(x.add(y))

def make_layers(c_in, c_out, repeat_times, is_downsample=False):
    blocks = []
    for i in range(repeat_times):
        if i == 0:
            blocks += [BasicBlock(c_in, c_out, is_downsample=is_downsample), ]
        else:
            blocks += [BasicBlock(c_out, c_out), ]
    return nn.Sequential(*blocks)
    
class Net(nn.Layer):
    def __init__(self, num_classes=751, reid=False):
        super(Net, self).__init__()
        # 3 128 64
        self.conv = nn.Sequential(
            nn.Conv2D(3, 64, 3, stride=1, padding=1),
            nn.BatchNorm2D(64),
            nn.ReLU(),
            # nn.Conv2d(32,32,3,stride=1,padding=1),
            # nn.BatchNorm2d(32),
            # nn.ReLU(inplace=True),
            nn.MaxPool2D(3, 2, padding=1),
        )
        # 32 64 32
        self.layer1 = make_layers(64, 64, 2, False)
        # 32 64 32
        self.layer2 = make_layers(64, 128, 2, True)
        # 64 32 16
        self.layer3 = make_layers(128, 256, 2, True)
        # 128 16 8
        self.layer4 = make_layers(256, 512, 2, True)
        # 256 8 4
        self.avgpool = nn.AvgPool2D((8, 4), 1)
        # 256 1 1 
        self.reid = reid
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1D(256),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = reshape(x, (x.shape[0], -1))  # x.view(x.size(0), -1)
        # B x 128
        if self.reid:
            x = div(x, x.norm(p=2, axis=1, keepdim=True))
            return x
        # classifier
        x = self.classifier(x)
        return x
```