# 🌄 Scene Classifier — Transfer Learning with ResNet18

A deep learning image classifier that identifies natural and urban scenes — **buildings, forest, glacier, mountain, sea, and street** — using transfer learning on a pretrained ResNet18 model.

## 🎯 Project Highlights
- Transfer learning with **PyTorch** and **ResNet18** (pretrained on ImageNet)
- Trained on the [Intel Image Classification dataset](https://www.kaggle.com/datasets/puneet6060/intel-image-classification) (~14,000 training images, 6 classes)
- ~90%+ test accuracy after fine-tuning
- Interactive **Streamlit** web app for real-time predictions
- Trained on Google Colab (free GPU)

## 🛠 Tech Stack
- Python, PyTorch, Torchvision
- Streamlit (web app)
- Google Colab (training)

## 📁 Project Structure
```
scene-classifier/
├── train_colab.ipynb     # Training notebook (run on Google Colab)
├── app.py                 # Streamlit demo app
├── requirements.txt
└── README.md
```

## 🚀 How to Run

### 1. Train the model (Google Colab)
- Open `train_colab.ipynb` in [Google Colab](https://colab.research.google.com/)
- Set runtime to GPU: `Runtime > Change runtime type > GPU`
- Run all cells — the dataset downloads automatically via `kagglehub`
- Download the trained `scene_classifier.pth` file at the end

### 2. Run the demo locally
```bash
git clone https://github.com/<your-username>/scene-classifier.git
cd scene-classifier
pip install -r requirements.txt
# place scene_classifier.pth in this folder
streamlit run app.py
```
Open `http://localhost:8501` and upload an image to see live predictions.

## 📊 Results
| Metric | Value |
|---|---|
| Train Accuracy | ~95% |
| Test Accuracy | ~90%+ |
| Epochs | 5 |
| Model | ResNet18 (transfer learning) |

## 📌 Future Improvements
- Fine-tune more layers instead of only the final classifier
- Add data augmentation experiments
- Deploy on Hugging Face Spaces / Streamlit Cloud

---
Built as a portfolio project to demonstrate transfer learning and end-to-end ML deployment.
## 📦 Model Download

The trained model file (`scene_classifier.pth`) is too large for GitHub. Download it here:

🔗 [Download scene_classifier.pth](https://drive.google.com/file/d/1n8DOX0IErI-qzMwf2q_nf4mwcImGpVgd/view?usp=sharing)

After downloading, place the file in the project root folder (same folder as `app.py`) before running the Streamlit app.