import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

st.set_page_config(page_title="Scene Classifier", page_icon="🌄", layout="centered")

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@st.cache_resource
def load_model():
    checkpoint = torch.load('scene_classifier.pth', map_location=DEVICE)
    class_names = checkpoint['class_names']

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)
    model.eval()
    return model, class_names

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

st.title("🌄 Scene Classifier")
st.write("Upload an image and the model will classify it as **buildings, forest, glacier, mountain, sea, or street**.")
st.caption("Transfer learning with ResNet18, trained on the Intel Image Classification dataset.")

model, class_names = load_model()

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=True)

    with st.spinner("Classifying..."):
        img_tensor = transform(image).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.softmax(outputs, dim=1)[0]
            top_prob, top_idx = torch.max(probs, 0)

    with col2:
        st.subheader("Prediction")
        st.success(f"**{class_names[top_idx].capitalize()}**")
        st.metric("Confidence", f"{top_prob.item()*100:.1f}%")

        st.write("All class probabilities:")
        for i, cname in enumerate(class_names):
            st.progress(probs[i].item(), text=f"{cname}: {probs[i].item()*100:.1f}%")
