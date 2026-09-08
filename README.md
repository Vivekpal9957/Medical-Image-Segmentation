# Medical Image Segmentation using U-Net

A deep learning project that performs medical image segmentation using the U-Net architecture. The model is trained to identify and segment regions of interest from medical images, making it useful for applications such as organ and tumor segmentation.

## Project Overview

Medical image segmentation is an important task in healthcare as it helps in accurately identifying different anatomical structures. In this project, a U-Net model has been implemented using PyTorch to perform pixel-wise segmentation on medical images.

The project covers the complete pipeline, including data preparation, model training, evaluation, and visualization of training performance.

## Features

- U-Net architecture implemented using PyTorch
- GPU acceleration using CUDA
- Training and validation pipeline
- Dice Score and IoU evaluation metrics
- Automatic model checkpoint saving
- Training history visualization
- Modular project structure

## Project Structure

```
Medical-Segmentation/
│── checkpoints/
│── results/
│── dataset.py
│── evaluate.py
│── main.py
│── model.py
│── train.py
│── utils.py
│── requirements.txt
└── README.md
```

## Technologies Used

- Python
- PyTorch
- NumPy
- Matplotlib
- OpenCV
- CUDA (GPU Training)

## Installation

Clone the repository

```bash
git clone https://github.com/your-username/Medical-Segmentation.git
```

Move into the project directory

```bash
cd Medical-Segmentation
```

Install the required packages

```bash
pip install -r requirements.txt
```

## Run the Project

```bash
python main.py
```

## Training Results

The model achieved strong segmentation performance during training.

| Metric | Value |
|---------|--------|
| Dice Score | 0.9999 |
| IoU Score | 0.9998 |

Training history is automatically saved in the `results` folder.

## Future Improvements

- Support multiple medical imaging datasets
- Add attention-based U-Net
- Improve data augmentation
- Deploy the model as a web application

## Author

**Vivek Pal**

B.Tech Student | Artificial Intelligence & Machine Learning

<img width="1892" height="912" alt="image" src="https://github.com/user-attachments/assets/11ed9958-1acc-466e-83d3-45bf6dcbc16b" />



<img width="2430" height="1830" alt="streamlit_ui_mockup" src="https://github.com/user-attachments/assets/31546d42-5448-4e67-822f-6588ba2cabd3" />

