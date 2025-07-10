# Airbag Detection System

A comprehensive web application for airbag detection and timing analysis using computer vision and machine learning.

## Features

- **Video Upload**: Drag and drop or browse to upload video files
- **AI Detection**: Uses YOLO models for object and name detection
- **Segmentation**: SAM (Segment Anything Model) for precise airbag segmentation
- **OCR**: Extract timing information from video frames
- **Real-time Processing**: Live progress updates with WebSocket-like polling
- **Results Management**: View, download, and manage processing results
- **Modern UI**: Clean, responsive React interface

## System Requirements

- Python 3.8+
- CUDA-compatible GPU (recommended)
- Node.js 16+ (for frontend development)
- At least 8GB RAM
- 10GB+ storage for models and processing

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/nnackpt/airbag-timing-detection.git
cd airbag-detection-system
```

### 2. Create Python Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download Required Models

Create a `models` directory in the project root and download the following models:

#### YOLO Models
- `YOLO_OBJECT.pt` - Object detection model
- `YOLO_NAME.pt` - Name recognition model

#### SAM Model
- Download `sam_vit_h_4b8939.pth` from [Segment Anything Model](https://github.com/facebookresearch/segment-anything)

#### OCR Model
- Place your OCR model in `models/ocr/` directory

```
models/
├── YOLO_OBJECT.pt
├── YOLO_NAME.pt
├── sam_vit_h_4b8939.pth
└── ocr/
    ├── config.json
    ├── pytorch_model.bin
    └── ...
```

### 5. Create Required Directories
```bash
mkdir -p uploads outputs screenshots models
```

## Usage

### Starting the Backend Server

```bash
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at `http://localhost:8000`

### Frontend Setup

For development, you can serve the React component using any static server or integrate it into your existing React application.

### API Documentation

Once the server is running, visit:
- API Documentation: `http://localhost:8000/docs`
- Alternative docs: `http://localhost:8000/redoc`

## API Endpoints

### Core Endpoints

- `POST /upload-video` - Upload video file
- `POST /process-video/{file_id}` - Start video processing
- `GET /task-status/{task_id}` - Get processing status
- `GET /tasks` - Get all tasks
- `POST /ocr` - Perform OCR on image
- `GET /download/{file_type}/{filename}` - Download files
- `DELETE /task/{task_id}` - Delete task

### Health Check

- `GET /health` - Check system status and model loading

## Configuration

### Processing Parameters

You can adjust these parameters in the UI or via API:

- **Confidence Threshold**: Detection confidence (0.1-1.0)
- **Motion Threshold**: Motion detection sensitivity
- **Start Frame**: Analysis start frame
- **End Frame**: Analysis end frame

### Model Paths

Update model paths in the configuration section of `main.py`:

```python
YOLO_OBJECT_MODEL_PATH = "models/YOLO_OBJECT.pt"
YOLO_NAME_MODEL_PATH = "models/YOLO_NAME.pt"
SAM_CHECKPOINT = "models/sam_vit_h_4b8939.pth"
OCR_MODEL_PATH = "models/ocr"
```

## Project Structure

```
airbag-detection-system/
├── main.py                 # FastAPI backend server
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── uploads/               # Uploaded video files
├── outputs/               # Processed video files
├── screenshots/           # Generated screenshots
├── models/                # AI models
│   ├── YOLO_OBJECT.pt
│   ├── YOLO_NAME.pt
│   ├── sam_vit_h_4b8939.pth
│   └── ocr/
└── frontend/              # Next frontend (if separated)
```

## Development

### Backend Development

The FastAPI backend uses:
- **FastAPI**: Modern web framework
- **OpenCV**: Computer vision operations
- **PyTorch**: Deep learning framework
- **Ultralytics**: YOLO implementation
- **Transformers**: OCR model handling

### Frontend Development

The Next frontend features:
- **Tailwind CSS**: Styling
- **Lucide Next**: Icons
- **Real-time Updates**: Progress tracking
- **File Management**: Upload/download handling

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size or use CPU mode
   - Close other GPU applications

2. **Model Loading Errors**
   - Verify model file paths
   - Check model file integrity
   - Ensure sufficient disk space

3. **Video Processing Fails**
   - Check video format compatibility
   - Verify video file isn't corrupted
   - Ensure sufficient processing time

4. **OCR Not Working**
   - Verify OCR model is properly installed
   - Check image quality and format
   - Ensure text is visible in images

### Performance Optimization

1. **Use GPU**: Ensure CUDA is properly installed
2. **Reduce Video Resolution**: For faster processing
3. **Adjust Confidence Thresholds**: Lower for more detections
4. **Batch Processing**: Process multiple videos sequentially

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review API documentation
3. Create an issue in the repository

## Acknowledgments

- **Meta**: Segment Anything Model (SAM)
- **Ultralytics**: YOLO implementation
- **Hugging Face**: Transformers library
- **FastAPI**: Web framework