# Historical Map Text Detection & Recognition

An end-to-end Deep Learning OCR pipeline designed to detect, recognize, and transcribe text from high-resolution historical maps.

## 📌 Project Overview

Historical maps contain valuable geographical and cultural data locked in image format. This project builds an automated pipeline to extract this text into a structured dataset.

The project initially explored a custom U-Net segmentation model, but ultimately transitioned to a state-of-the-art PaddleOCR pipeline due to its superior performance on complex map text (multi-orientation, warped text, variable spacing, etc.).

## 🚀 Key Features

- **High-Accuracy OCR**: Utilizes PaddleOCR's DBNet for detection and CRNN for recognition
- **Robust Pre-processing**: Handles high-resolution historical map scans
- **Batch Processing**: Fully automated extraction across directories
- **Structured Output**: Exports text, bounding boxes, and confidence scores to CSV
- **Proven Performance**: Achieved a mean confidence of 96.16% on the final filtered dataset

## 📂 Repository Structure

```
.
├── final_inference.ipynb          # MAIN: PaddleOCR pipeline for batch processing
├── map_ocr_results_CLEANED.csv    # FINAL OUTPUT: High-quality filtered text dataset
├── benchmark_paddle_output.png    # VISUAL: Example detection output with bboxes
├── README.md
├── experiments/                   # ARCHIVE: U-Net segmentation experiments
│   ├── gpu_map_ocr_test.ipynb     # U-Net Training Notebook
│   ├── unet_resnet18_50_epochs.pth
│   └── unet_validation_detection_sample.png
└── Rumsey_Map_OCR_Data/           # Training data
    └── rumsey/
        ├── icdar24-train-png/
        │   ├── annotations.json
        │   └── train_images/
        └── icdar24-val-png/
            ├── annotations.json
            └── val_images/
```

## 🛠️ Installation & Requirements

This project requires a Python environment with GPU support (recommended).

### Dependencies

- `paddlepaddle-gpu` (or `paddlepaddle` for CPU)
- `paddleocr>=2.7.0`
- `opencv-python`
- `numpy<2.0` (Critical for compatibility)
- `pandas`

### Setup

```bash
# Create and activate environment
conda create -n map_ocr python=3.10
conda activate map_ocr

# Install specific versions for stability
pip install paddlepaddle-gpu
pip install "paddleocr==2.7.3"
pip install "numpy<2.0.0"
pip install opencv-python matplotlib pandas
```

## 💻 Usage

To run the full OCR extraction pipeline on your own map images:

1. **Prepare Images**: Place your `.png` or `.jpg` map images into a directory (e.g., `data/val_images/`)
2. **Open Notebook**: Launch `final_inference.ipynb`
3. **Configure Path**: Set the `VAL_ROOT` variable to your image directory
4. **Run All Cells**: The pipeline will iterate through every image and save results to the official CSV

## 📊 Results & Benchmarks

The final dataset was filtered using a 0.80 confidence threshold to remove noisy readings.

| Metric | Initial Readings | Final Cleaned Dataset |
|--------|------------------|----------------------|
| Total Text Regions | 2,767 | 2,382 (86.09% Retained) |
| Mean Confidence (Quality Score) | 0.9179 | 0.9616 |
| Median Confidence | 0.9761 | 0.9850 |
## 🔬 Experiments (U-Net Architecture)

The `experiments/` directory contains the initial research using a custom U-Net with a ResNet18 backbone. This work is preserved for documentation, proving the necessity of benchmarking against SOTA models for complex data.

## 🔮 Future Work

- **Search Engine**: Build a tool to search for specific geographical features using the CSV coordinates
- **Georeferencing**: Map extracted pixel coordinates to real-world GPS coordinates for GIS applications
- **Text Density Analysis**: Visualize "hotspots" of text on different maps