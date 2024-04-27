import torch.nn as nn

class LeNet(nn.Module):
    def __init__(self, num_classes, in_channels):
        super(LeNet, self).__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels

        self.conv1 = nn.Conv2d(in_channels, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 9 * 9, 120)  # 5*5 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
    
    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = nn.functional.max_pool2d(nn.functional.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = nn.functional.max_pool2d(nn.functional.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

if __name__ == "__main__":
    import torch
    tensor = torch.rand((16,1,48,48), device="cuda:0")
    model = LeNet(8, 1)
    model = model.cuda("cuda:0")
    out = model(tensor)
    import torchview
    graph = torchview.draw_graph(model, input_data=tensor)
    image = graph.visual_graph.render("lenet_torchview")