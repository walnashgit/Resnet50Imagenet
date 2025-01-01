import torch
import gradio as gr
import torchvision.transforms as transforms
from ResnetNetwork import ResNet50LightningModule
import torch.nn.functional as F

# Load class mappings and limit to 2 names per class
with open('LOC_synset_mapping.txt', 'r') as f:
    class_names = []
    for line in f.readlines():
        # Split off the synset ID and get the names
        names = line.strip().split(' ', 1)[1]
        # Split on comma and take first two names
        names = ', '.join(names.split(',')[:2])
        class_names.append(names)

# Initialize model and load weights
def load_model(checkpoint_path):
    model = ResNet50LightningModule(lr_dataloader=None, lr_finder=False)
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225])
])

def predict(image):
    # Preprocess the image
    img = transform(image).unsqueeze(0)
    
    # Get model predictions
    with torch.no_grad():
        outputs = model(img)
        probabilities = F.softmax(outputs, dim=1)
        
    # Get top 5 predictions
    top5_prob, top5_indices = torch.topk(probabilities, 5)
    
    # Create results dictionary with probabilities as percentages (0-100)
    results = {class_names[idx.item()]: prob.item()
               for prob, idx in zip(top5_prob[0], top5_indices[0])}

    return results

# Create Gradio interface
iface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=5),
    title="ResNet50 Image Classifier",
    description="Upload an image to get the top 5 predictions from the ImageNet-trained ResNet50 model.",
    examples=[
        ["examples/dog.jpg"],
        ["examples/cat.jpg"],
        ["Resnet50/tabla02.jpeg"]
    ],
    theme=gr.themes.Soft()
)

if __name__ == "__main__":
    # Load the model
    model = load_model("checkpoints/resnet50-epoch=39-val_acc=73.55.ckpt")
    
    # Launch the app
    iface.launch(share=True) 