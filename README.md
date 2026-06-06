# FOTS-based End-to-End Text Detection and Recognition

## Project Overview
This repository contains a high-performance, unified pipeline for **Text Detection and Recognition** in natural scene images. It implements the **FOTS (Fast Oriented Text Spotting)** architecture, which treats detection and recognition as a single, joint optimization problem. This approach significantly reduces latency and improves accuracy by sharing convolutional features between the detection and recognition branches.

## What is this Project?
This project is an end-to-end OCR application that:
- **Unified Spotting:** Detects and recognizes text in a single forward pass using a shared feature extractor.
- **Oriented Text Handling:** Capable of detecting text at any rotation angle using **RBOX (Rotated Bounding Box)** geometry.
- **Interactive Web UI:** Features a **Streamlit** dashboard for real-time testing and visualization of the spotting pipeline.

## How it was done (Deep Technical Details)
- **Neural Network Architecture:**
    - **Backbone:** A shared CNN (Inception-style or ResNet) that extracts multi-scale features.
    - **Feature Merging Branch:** A custom Keras layer (`feature_merging_branch`) that merges features from different stages of the backbone using bilinear **UpSampling2D** and **Concatenate** layers to preserve spatial and semantic information.
    - **Detection Branch:** Predicts a per-pixel score map and a 5-channel geometry map (RBOX: distances to 4 edges + rotation angle).
    - **Recognition Branch:** A sequence-to-sequence model that uses **ROIRotate** to extract oriented features and an RNN (LSTM) to decode the text.
- **Loss Functions:**
    - **Detection Loss:** A multi-task loss combining **Dice Loss** (for score map), **IOU Loss** (for bounding box regression), and **Angle Loss** (cosine similarity for rotation).
    - **Recognition Loss:** **CTC (Connectionist Temporal Classification) Loss**, allowing for training on unaligned text sequences.
- **Inference Pipeline:**
    - **Restoration Logic:** `restore_rectangle_rbox` converts the 5-channel geometry map into actual coordinates.
    - **Decoding:** Uses `tf.keras.backend.ctc_decode` along with a pre-trained **Pickle-based Tokenizer** to convert model predictions into human-readable strings.
- **Optimization:** Models are converted to **TensorFlow Lite (TFLite)** for high-speed, edge-ready inference.

## Tech Stack
- **Deep Learning Framework:** TensorFlow 2.x, Keras
- **Web UI:** Streamlit
- **Computer Vision:** OpenCV (`cv2`), PIL
- **Geometry & Math:** NumPy, Shapely (for polygon and IOU calculations)
- **Model Formats:** H5 (Keras), TFLite (TensorFlow Lite)
- **Data Handling:** Pandas, Pickle

## Key Features
- **Real-time Scene Text Spotting:** From raw image input to localized and recognized text output.
- **Advanced Loss Implementation:** Custom implementation of Dice and IOU losses for robust training.
- **Comprehensive Notebooks:**
    - `FOTS_data_prepare.ipynb`: Detailed dataset preparation.
    - `pipeline.ipynb`: Step-by-step walkthrough of the spotting and recognition logic.

## File Structure
- `develop.py`: The entry point for the Streamlit web application.
- `ml_models.py`: The core engine containing custom layers (`feature_merging_branch`), custom losses, and the `inferencePipeline_lite`.
- `pipeline.ipynb`: Interactive demonstration of the full OCR pipeline.
- `tokenizer.pickle`: Pre-trained mapping for text recognition.

## Local Setup
1.  **Clone the repository:**
    ```bash
    git clone [repository-url]
    ```
2.  **Install dependencies:**
    ```bash
    pip install streamlit tensorflow opencv-python numpy shapely
    ```
3.  **Run the application:**
    ```bash
    streamlit run develop.py
    ```
