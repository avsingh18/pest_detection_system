# pest_detection_system
# Plant Disease Detection System 🌿

A deep learning-based system for detecting diseases in plant leaves using TensorFlow and Computer Vision.

## Overview 🔍

This project implements a Convolutional Neural Network (CNN) to identify various plant diseases from leaf images. It can help farmers and gardeners quickly diagnose plant diseases for better crop management.

## Features ⭐

- Real-time plant disease detection
- Support for multiple image formats (JPG, JPEG, PNG, WEBP)
- Interactive command-line interface
- Confidence score for predictions
- Detailed interpretation of results
- Multi-class disease classification

## Installation 🛠️

1. Clone the repository:
```bash
git clone https://github.com/avsingh18/pest_detection_system.git
cd pest_detection_system
```

2. Install required packages:
```bash
pip install tensorflow[and-cuda]==2.19.0
pip install pillow numpy matplotlib
```

## Usage 📋

### Training the Model

1. Run the training script:
```bash
python plant_disease_detection.py
```

This will:
- Download the PlantVillage dataset
- Train the CNN model
- Save the trained model and class names

### Making Predictions

1. Run the prediction script:
```bash
python predict.py
```

2. When prompted, enter the path to your image file
3. View the prediction results, including:
   - Detected disease/condition
   - Confidence level
   - Result interpretation

## Project Structure 📁

```
pest_detection_system/
├── plant_disease_detection.py   # Training script
├── predict.py                   # Prediction script
├── output/                      # Generated files
│   ├── best_model.h5           # Trained model
│   ├── class_names.json        # Disease classes
│   └── training_history.png    # Training plots
└── README.md
```

## Model Details 🧮

- Architecture: CNN with BatchNormalization
- Input Size: 224x224x3
- Output: Multi-class classification
- Training Dataset: PlantVillage

## Requirements 📝

- Python 3.8 or higher
- TensorFlow 2.19.0
- CUDA support (optional, for GPU acceleration)
- PIL (Python Imaging Library)
- NumPy
- Matplotlib

## Performance 📊

- Training Accuracy: ~95%
- Validation Accuracy: ~92%
- Supports multiple plant species and diseases

## Troubleshooting 🔧

Common issues and solutions:

1. **Import Error**: Make sure all required packages are installed:
```bash
pip install tensorflow[and-cuda]==2.19.0 pillow numpy matplotlib
```

2. **GPU Issues**: If experiencing GPU problems, try using CPU-only version:
```bash
pip install tensorflow-cpu==2.19.0
```

3. **Memory Error**: Reduce batch size in training script

## Contributing 🤝

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## License 📄

This project is licensed under the MIT License - see the LICENSE file for details.

## Author ✍️

- **Avadhesh Singh** - [@avsingh18](https://github.com/avsingh18)

## Acknowledgments 🙏

- PlantVillage Dataset
- TensorFlow Team
- Open Source Community

## Last Updated 📅

2025-07-05

## Contact 📬

For any queries or suggestions, please open an issue in the repository.

---
Made with ❤️ by Avadhesh Singh