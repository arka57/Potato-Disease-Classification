import io
import torch 
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

#For loading the model
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet,self).__init__()
        self.cnn=nn.Sequential(
            nn.Conv2d(3,6,(5,5)),
            nn.AvgPool2d(2,stride=2),
            nn.ReLU(),
            nn.Conv2d(6,16,(5,5)),
            nn.AvgPool2d(2,stride=2),
            nn.ReLU(),
            nn.Conv2d(16,8,(5,5)),
            nn.AvgPool2d(2,stride=2),
            nn.ReLU(),
            nn.Conv2d(8,4,(5,5)),
            nn.AvgPool2d(2,stride=2),
            nn.ReLU()

        )
        self.fc=nn.Sequential(
            
            nn.Linear(400,200),

            nn.ReLU(),
            nn.Linear(200,80),
            nn.ReLU(),
            nn.Linear(80,3)
        )
    def forward(self,x):
        #print(x.shape)
        x=self.cnn(x)
        #print(x.shape)
        #print(x.size(0))
        x=x.view(x.size(0),-1)
        #print(x.shape)
        x=self.fc(x)
        #print(x.shape)
        return x

#Class Label
class_label=['Potato___Early_blight', 'Potato___Late_blight','Potato___healthy']


loaded_model=LeNet()
loaded_model.load_state_dict(torch.load("model1.th", map_location=torch.device('cpu')) )
loaded_model.eval()


def transform_images(image_bytes):

    transform = transforms.Compose([transforms.RandomResizedCrop(224),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
                                    

    image = Image.open(io.BytesIO(image_bytes))#reading the image bytes by bytes
    return transform(image).unsqueeze(0) #transforming the image and doing unsqueeze to add one more dimension. 3,224,224-->1,3,224,224. This is for handling batch size=1

def predict(image):
    output=loaded_model.forward(image)
    _,predicted_label=torch.max(output,1)
    return class_label[predicted_label]
