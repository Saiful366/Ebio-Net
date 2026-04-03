# 🌿 Ebio-Net 

A powerful, high-performance AI classification pipeline for detection and monitoring of disease types and spread. This application leverages an optimized **EfficientNet** neural engine to provide scalable high-throughput plant disease type identification with 90%+ target accuracy.

![Ebio-Net](Image/Image.png)

## 🚀 Key Features

*   **Precision Neural Scan**: High-fidelity analysis for single plant leaves with automated image labeling.
*   **Mass Pipeline Processing**: Bulk classification via folder path input or manual multi-file upload.
*   **Visual Data Proofing**: Every analyzed asset is generated with a burned-in classification tag (Type & Confidence %).
*   **Comprehensive Telemetry**: Export detailed prediction matrices for every file in a dataset to the standard CSV format.
*   **Luxury Enterprise UI**: A modern, glassmorphic interface designed for clarity and professional presentation.

## 🛠️ Quick Start Guide

### 1. Get the App
Clone the repository and navigate into the App folder:
```bash
git clone https://github.com/Saiful366/Ebio-Net.git
cd Ebio-Net/App
```
> Or download the ZIP from GitHub and open the `App` folder in your terminal.

### 2. Environment Setup
Create and activate a dedicated conda environment (Python 3.11 recommended for TensorFlow compatibility):
```bash
conda create -n modelapp python=3.11
conda activate modelapp
pip install -r requirements.txt
```

### 3. Launching the Engine
Execute the following command to boot the local analysis server:
```bash
streamlit run app.py
```
The application will launch at `http://localhost:8501`.

## 💻 Operational Modes

### 🎯 Precision Analysis (Single)
1. Navigate to the **PRECISION SCAN** tab.
2. Select or drop a high-resolution leaf sample.
3. Click `EXECUTE NEURAL SCAN`.
4. Review the **Intelligence Report** featuring annotated visual proof and the full probability map.

### 🚀 Batch Pipeline (Folder or Upload)
1. Navigate to the **BATCH PIPELINE** tab.
2. **Option A — Directory Path**: Paste the full local folder path (e.g. `/Users/saiful/Downloads/images`) into the directory input field and click `CONFIRM DIRECTORY`.
3. **Option B — Manual Upload**: Select multiple images using the file uploader.
4. Click `START PIPELINE PROCESSING` to begin the mass-scan.
5. Download the consolidated **CSV Report** once the progress bar reaches 100%.

## 📁 System Architecture
*   `app.py`: Main application core and UI controller.
*   `run.py`: Helper script to launch the application.
*   `restored_best_model_88.keras`: The optimized Deep Learning weight file.
*   `requirements.txt`: Dependency specifications.
*   `README.md`: This technical documentation.

## ⚠️ Engine Specifications
*   **Neural Core**: Keras Optimized EfficientNetB0
*   **Input Resolution**: 224x224 (Auto-Standardized)
*   **Preprocessing**: Neural Spectral Normalization (v3.6)
*   **Classification Depth**: 6 Distinct Plant Types
*   **Python**: 3.11 (conda environment: `modelapp`)

## 🔧 Troubleshooting

### `ModuleNotFoundError: No module named 'tensorflow'`
TensorFlow is not installed in your active environment. Make sure you have activated the correct conda environment and install TensorFlow manually:
```bash
conda activate modelapp
pip install tensorflow
```
> **Note**: TensorFlow supports Python 3.9–3.11. If you are using Python 3.12 or 3.13, create the environment with Python 3.11 explicitly:
> ```bash
> conda create -n modelapp python=3.11
> conda activate modelapp
> pip install -r requirements.txt
> ```

### `ModuleNotFoundError: No module named 'streamlit'`
```bash
pip install streamlit
```

### `ModuleNotFoundError: No module named 'cv2'`
```bash
pip install opencv-python-headless
```

### App opens but model fails to load
Ensure `restored_best_model_88.keras` is placed inside the `App` folder alongside `app.py`:
```
App/
├── app.py
├── restored_best_model_88.keras
└── requirements.txt
```

### Drag and drop not working in browser
Use the **Browse files** button instead. This is a known browser limitation. The app works best in **Google Chrome** or **Firefox**.

## 📄 License

This project is licensed under the **MIT License**.

```
MIT License

Copyright (c) 2026 Ebio-Net Systems

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
```

## 📚 Citation

If you use Ebio-Net in your research or project, please cite it as:

```bibtex
@software{ebionet2026,
  author       = {Saiful Islam},
  title        = {Ebio-Net: AI-Powered Plant Disease Classification Pipeline},
  year         = {2026},
  publisher    = {GitHub},
  url          = {https://github.com/Saiful366/Ebio-Net},
  version      = {v3.6}
}
```

Or in plain text:

> Saiful Islam. (2026). *Ebio-Net: AI-Powered Plant Disease Classification Pipeline* (v3.6). GitHub. https://github.com/Saiful366/Ebio-Net

---
*© 2026 Ebio-Net Systems • v3.6 Industrial Core*
