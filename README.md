# FOTS-based Text Detection and Recognition

## Project Overview
This repository contains a specialized implementation of a **Text Detection and Recognition** system, primarily focusing on the **FOTS (Fast Oriented Text Spotting)** architecture. The project aims to provide a unified and efficient solution for extracting text from natural images, even those with complex orientations and backgrounds.

## What is this Project?
The project provides a complete workflow for advanced OCR (Optical Character Recognition):
- **Integrated Spotting:** Leverages a unified pipeline that combines detection and recognition for better performance compared to separate multi-stage approaches.
- **Interactive UI:** A built-in **Streamlit** application allows users to upload images and visualize the detected text regions and recognition results instantly.
- **Data Engineering:** Includes specialized notebooks for preparing and augmenting datasets specifically for oriented text spotting tasks.

## How it was done
- **Model Architecture:** Explores the FOTS framework which uses a shared feature extractor and specialized branches for text detection and ROI (Region of Interest) recognition.
- **Development Workflow:**
  - `FOTS_data_prepare.ipynb`: Manages the complex task of preparing training data, including handling bounding box coordinates and text labels.
  - `pipeline.ipynb`: Documents the integration and testing of the end-to-end spotting pipeline.
  - `ml_models.py`: Encapsulates the inference logic, making it easy to integrate into applications.
- **Optimization:** The system is designed to be fast, suitable for real-time or near-real-time applications.

## Why it was done
- To implement and evaluate state-of-the-art text spotting architectures.
- To create a tool that can accurately read text in natural scenes (e.g., street signs, product labels) regardless of orientation.
- To simplify the deployment of advanced OCR systems through a user-friendly web interface.

## Tech Stack
- **Language:** Python
- **Web Framework:** Streamlit
- **Deep Learning:** TensorFlow / Keras
- **Computer Vision:** OpenCV (`cv2`)
- **Data Handling:** NumPy, Pandas

## File Structure
- `develop.py`: The entry point for the Streamlit web application.
- `ml_models.py`: Contains the core inference and model loading logic.
- `FOTS_data_prepare.ipynb`: Notebook for dataset preparation and preprocessing.
- `pipeline.ipynb`: Comprehensive notebook showing the detection and recognition pipeline in action.

## Local Setup
1.  **Clone the repository:**
    ```bash
    git clone [repository-url]
    ```
2.  **Install dependencies:**
    ```bash
    pip install streamlit opencv-python tensorflow numpy
    ```
3.  **Run the application:**
    ```bash
    streamlit run develop.py
    ```
