import torch
from PIL import Image
from torchvision import transforms


class ClassificationModel:

    def __init__(self, path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.load(path, map_location=self.device)
        self.model.eval()
        self.classes = ['Audi', 'Mitsubishi', 'Opel', 'Toyota', 'Volkswagen', 'bmw', 'bus', 'chevrolet', 'citroen',
                        'daewoo', 'ford', 'gaz', 'hyundai', 'jeep', 'kia', 'lada', 'lexus', 'mercedes', 'nissan',
                        'peugeot', 'renault', 'skoda', 'truck', 'vans']

    def predict_image(self, image):
        test_transforms = transforms.Compose([transforms.Resize((244, 244)),
                                              transforms.CenterCrop(224),
                                              transforms.ToTensor(),
                                              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        image_tensor = test_transforms(Image.fromarray(image))  # .float()
        image_tensor = image_tensor.unsqueeze_(0)
        # input = torch.Tensor(image_tensor)
        input_model = image_tensor.to(self.device)
        output = self.model(input_model)
        index = output.data.cpu().numpy().argmax()
        return self.classes[index]
