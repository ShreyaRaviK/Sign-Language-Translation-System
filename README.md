# SIGNSIGHT: Sign Language Translator

## Overview

SIGNSIGHT is an AI-powered web application that translates sign language gestures into text and speech in real-time. The system uses computer vision techniques to detect hand movements and machine learning models to classify signs into letters, numbers, and words. It integrates with Google's Gemini API for natural language processing and text-to-speech functionality to provide audio output.

The application supports three main modes:
- **Letter Mode**: Recognizes individual letters from the American Sign Language (ASL) alphabet
- **Number Mode**: Recognizes numbers 0-9 in ASL
- **Word Mode**: Recognizes a vocabulary of 126 common words/phrases

## Features

- **Real-time Translation**: Live video processing for immediate sign recognition
- **Multiple Recognition Modes**: Support for letters, numbers, and words
- **Video Upload Support**: Process pre-recorded sign language videos
- **Text-to-Speech**: Automatic audio output of translated text
- **Web Interface**: User-friendly Flask-based web application
- **Model Integration**: Uses trained PyTorch models for accurate classification
- **LLM Enhancement**: Leverages Google Gemini API for contextual understanding
- **Hand Tracking**: Advanced hand landmark detection using MediaPipe

## Technologies Used

- **Backend**: Python Flask
- **Computer Vision**: OpenCV, MediaPipe
- **Machine Learning**: PyTorch, scikit-learn
- **AI Integration**: Google Gemini API
- **Text-to-Speech**: Google Text-to-Speech (gTTS)
- **Frontend**: HTML, CSS, JavaScript
- **Data Processing**: NumPy, SciPy

## Installation

### Prerequisites

- Python 3.8 or higher
- Virtual environment (recommended)

### Setup Steps

1. **Clone the repository** (if applicable) or navigate to the project directory.

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment**:
   - On Windows:
     ```bash
     venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     source venv/bin/activate
     ```

4. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

5. **Set up environment variables**:
   - Create a `.env` file in the root directory
   - Add your Google Gemini API key:
     ```
     GEMINI_API_KEY=your_api_key_here
     ```

6. **Ensure model files are in place**:
   - Place trained model files in the `models/` directory:
     - `best_lightweight.pt` (word recognition model)
     - `classify_letter_model.p` (letter classification model)
     - `classify_number_model.p` (number classification model)
     - `label_map.json` (word labels)
     - `lightweight_tgcn.onnx` (ONNX model for inference)

## Usage

### Running the Application

1. **Start the Flask server**:
   ```bash
   python app.py
   ```

2. **Open your web browser** and navigate to `http://localhost:5000`

### Using the Application

- **Home Page**: Select between Live Translation or Video Upload modes
- **Live Translation**: Choose a mode (Letter, Number, or Word) and use your webcam for real-time sign recognition
- **Video Upload**: Upload a video file for processing and translation
- **Audio Output**: Translated text is automatically converted to speech

### API Endpoints

- `GET /`: Home page
- `GET /live`: Live translation interface
- `GET /upload`: Video upload interface
- `POST /upload`: Process uploaded video
- `GET /video_feed`: Webcam video stream for live mode
- Various WebSocket endpoints for real-time communication

## Project Structure

```
├── app.py                      # Main Flask application
├── requirements.txt            # Python dependencies
├── test_*.py                   # Test files for various components
├── models/                     # Trained ML models and labels
│   ├── best_lightweight.pt     # Word recognition PyTorch model
│   ├── classify_letter_model.p # Letter classification model
│   ├── classify_number_model.p # Number classification model
│   ├── label_map.json          # Word label mappings
│   └── lightweight_tgcn.onnx   # ONNX model for inference
├── static/                     # Static web assets
│   └── style.css               # CSS stylesheets
├── templates/                  # HTML templates
│   ├── index.html              # Home page
│   ├── live.html               # Live translation page
│   └── upload.html             # Video upload page
├── uploads/                    # Directory for uploaded files
├── processed/                  # Directory for processed videos
└── utils/                      # Utility functions
    └── utils.py                # Helper functions
```

## Model Training

The application uses pre-trained models for sign recognition. To train new models:

1. Prepare your dataset with sign language gestures
2. Use the provided training scripts (if available) or implement custom training
3. Save models in the appropriate format (PyTorch .pt files or pickle .p files)
4. Update the `label_map.json` for word classifications

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- MediaPipe for hand tracking capabilities
- Google for Gemini API and Text-to-Speech services
- PyTorch community for machine learning framework
- OpenCV for computer vision utilities

## Contact

For questions or support, please open an issue in the repository or contact the development team.