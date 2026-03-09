import torch
import torch.nn as nn

class BaselineCNN(nn.Module):

    def __init__(self, input_channels : int = 3, image_size : int = 224, conv_channels : list = [32, 64, 128], 
                 kernel_sizes : list = [3, 3, 3], fc_layers : list = [128], dropout=0.0,
                 num_classes : int = 10):
        
        '''
        Initializes the BaselineCNN model.
        Conv2d with ReLU and MaxPool2d -> Flatten -> Fully Connected layers with ReLU and Dropout -> Output

        Args:
            input_channels (int): Number of input channels (default: 3 for RGB images).
            image_size (int): Size of the input images (default: 224 for 224x224 images).
            conv_channels (list): List of output channels for each convolutional layer.
            kernel_sizes (list): List of kernel sizes for each convolutional layer.
            fc_layers (list): List of output features for each fully connected layer.
            dropout (float): Dropout rate for regularization (default: 0.5).
            num_classes (int): Number of output classes for classification (default: 10).
        '''
        
        super().__init__()

        convolutional_layers = []
        in_channels = input_channels

        #constructing convolutional layers
        for output_channels, kernel_size in zip(conv_channels, kernel_sizes):
            convolutional_layers.append(nn.Conv2d(in_channels, output_channels, kernel_size=kernel_size, padding=1))
            convolutional_layers.append(nn.BatchNorm2d(output_channels))
            convolutional_layers.append(nn.ReLU())
            convolutional_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            in_channels = output_channels

        self.conv = nn.Sequential(*convolutional_layers)
        self.global_pool = nn.AdaptiveAvgPool2d((1,1))

        conv_output_features = in_channels
        
        fully_connected_layers = []
        in_features = conv_output_features

        #constructing fully connected layers
        for output_features in fc_layers:
            fully_connected_layers.append(nn.Linear(in_features, output_features))
            fully_connected_layers.append(nn.ReLU())
            if dropout>0:
                fully_connected_layers.append(nn.Dropout(dropout))
            in_features = output_features
        
        #final output layer -> returning logits for each class
        fully_connected_layers.append(nn.Linear(in_features, num_classes))

        self.fc = nn.Sequential(*fully_connected_layers)


    def forward(self, x):

        '''
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes) containing the logits for each class.
        '''

        x = self.conv(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    



        

