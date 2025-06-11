# Multi-VLM Processing Server

A FastAPI-based application that processes images using multiple Vision Language Models (VLMs) to generate comprehensive descriptions. This project integrates three different VLM models: Kimi VL, Qwen 2.5 VL, and SmolVLM to provide diverse perspectives on image analysis.

## Features

- **Multi-Model Processing**: Utilizes three different VLM models for comprehensive image analysis
- **REST API**: FastAPI-based web service for easy integration
- **Base64 Image Support**: Accept base64 encoded images for seamless integration
- **Concurrent Processing**: Processes images with all three models and returns combined results
- **Fashion & Lifestyle Focus**: Optimized for product descriptions in fashion and lifestyle contexts
- **Error Handling**: Robust error handling with appropriate HTTP status codes
- **Automatic File Management**: Temporary image files are automatically managed and cleaned up

## Supported Models

1. **Kimi VL A3B Instruct** - Advanced vision-language model for detailed image understanding
2. **Qwen 2.5 VL** - Alibaba's multimodal large language model
3. **SmolVLM** - Efficient vision-language model for image summarization

## Project Structure

```
.
├── main.py                              # Standalone script for testing models
├── requirements.txt                     # Python dependencies
├── README.md                           # Project documentation
└── src/
    ├── configs/
    │   └── __init__.py
    ├── core/
    │   ├── __init__.py
    │   └── machine_learning/
    │       ├── __init__.py
    │       ├── kimi_vl_A3B_instruct_summarization.py
    │       ├── qwen2_5vl_img_summarization.py
    │       └── smolvlm_img_summarization.py
    ├── server/
    │   ├── __init__.py
    │   └── rest_api/
    │       ├── __init__.py
    │       └── app.py                   # FastAPI application
    └── utils/
        ├── __init__.py
        └── temp.jpg                     # Temporary location for base64 images
```

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/aimlBysoham/Multi-VLM-Processing-Server.git
   cd Multi-VLM-Processing-Server
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   conda create -n vlm-env python=3.10
   conda activate vlm-env
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running the FastAPI Server on Terminal (Debug Version)

Start the web server on port 7090:

```bash
python src/server/rest_api/app.py
```

### Running the FastAPI Server on Terminal (Hosting it independently)
Run the below command inside the folder of Rest API Server, to get live api logs and api process id saved for you. Even if you close the terminal, the api service stays up. 
```bash
cd src/server/rest_api/
nohup python -m uvicorn app:app   --host 0.0.0.0   --port 7090   > api.log 2>&1 &
echo $! > api.pid
```

The API will be available at `http://localhost:7090`

### API Documentation

Once the server is running, visit:
- **Interactive API docs**: `http://localhost:7090/docs`
- **ReDoc documentation**: `http://localhost:7090/redoc`

### API Endpoint

**POST** `/process-image`

**Request Body:**
```json
{
    "image_base64": "base64_encoded_image_data",
    "prompt_text": "Describe me the product in the image."
}
```

**Response:**
```json
{
    "kimi_description": "Detailed description from Kimi VL model",
    "qwen_description": "Detailed description from Qwen 2.5 VL model", 
    "smolvlm_description": "Detailed description from SmolVLM model"
}
```

**Example cURL request:**
```bash
curl -X POST "http://localhost:7090/process-image" \
     -H "Content-Type: application/json" \
     -d '{
       "image_base64": "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAYEBQYFBAYGBQYHBwYIChAKCgkJChQODwwQFxQYGBcUFhYaHSUfGhsjHBYWICwgIyYnKSopGR8tMC0oMCUoKSj...",
       "prompt_text": "Describe me the product in the image."
     }'
```

**JavaScript/Frontend Example:**
```javascript
// Convert image file to base64
const fileToBase64 = (file) => {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.readAsDataURL(file);
        reader.onload = () => resolve(reader.result.split(',')[1]); // Remove data URL prefix
        reader.onerror = error => reject(error);
    });
};

// Send image for processing
const processImage = async (imageFile, prompt) => {
    try {
        const base64Image = await fileToBase64(imageFile);
        
        const response = await fetch('http://localhost:7090/process-image', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                image_base64: base64Image,
                prompt_text: prompt
            })
        });
        
        const result = await response.json();
        console.log(result);
    } catch (error) {
        console.error('Error processing image:', error);
    }
};
```

### Base64 Image Format Support

The API accepts base64 encoded images in the following formats:
- **Raw base64**: Direct base64 encoded image data
- **Data URL format**: `data:image/jpeg;base64,{base64_data}`
- **Supported image types**: JPEG, PNG, GIF, WebP, and other common formats

The server automatically handles:
- Data URL prefix removal
- Base64 decoding and validation
- Temporary file creation at `src/utils/temp.jpg`
- Automatic cleanup of temporary files

### Running the Standalone Script

For testing individual models or batch processing:

```bash
python main.py
```

This will process the sample image (`src/utils/temp.jpg`) with all three models and display the results.

## Model Processing Flow

1. **Base64 Decoding**: The base64 image data is decoded and saved to a temporary file
2. **Model Loading**: Each VLM processor loads its respective model
3. **Image Processing**: The temporary image file is processed with the provided prompt
4. **Description Generation**: Each model generates a description based on the image and prompt
5. **Model Unloading**: Models are unloaded to free up memory
6. **File Cleanup**: Temporary image files are automatically removed
7. **Response Compilation**: All descriptions are combined and returned

## Error Handling

The API includes comprehensive error handling:

- **400 Bad Request**: When base64 image data is invalid or cannot be decoded
- **500 Internal Server Error**: For any processing errors during model execution
- **File Management**: Automatic cleanup of temporary files even on errors

## Performance Considerations

- Models are loaded and unloaded for each request to manage memory usage
- Base64 decoding adds minimal overhead compared to model processing time
- Processing time varies depending on image size and model complexity
- Temporary files are automatically cleaned up to prevent disk space issues
- Consider implementing model caching for production deployments with high traffic

## Development

### Adding New Models

1. Create a new processor class in `src/core/machine_learning/`
2. Implement the required methods: `load_model()`, `process()`, and `unload_model()`
3. Import and initialize the processor in both `app.py` and `main.py`
4. Add the model processing logic to the endpoint

### Customizing Prompts

The default prompt focuses on fashion and lifestyle products. You can customize prompts by:
- Modifying the default prompt in `main.py`
- Sending custom prompts via the API endpoint
- Creating prompt templates for different use cases

### Integration Examples

**Python Client:**
```python
import base64
import requests

def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Process image
base64_image = encode_image_to_base64("path/to/image.jpg")
response = requests.post(
    "http://localhost:7090/process-image",
    json={
        "image_base64": base64_image,
        "prompt_text": "Describe this product in detail."
    }
)
result = response.json()
```

**Node.js Client:**
```javascript
const fs = require('fs');
const axios = require('axios');

const encodeImageToBase64 = (imagePath) => {
    const image = fs.readFileSync(imagePath);
    return Buffer.from(image).toString('base64');
};

const processImage = async (imagePath, prompt) => {
    const base64Image = encodeImageToBase64(imagePath);
    
    try {
        const response = await axios.post('http://localhost:7090/process-image', {
            image_base64: base64Image,
            prompt_text: prompt
        });
        
        console.log(response.data);
    } catch (error) {
        console.error('Error:', error.response?.data || error.message);
    }
};
```

## Requirements

- Python 3.10+
- FastAPI
- Pydantic
- Uvicorn
- Additional model-specific dependencies (see `requirements.txt`)

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Moonshot.ai Kimi VL team for the A3B Instruct model
- Alibaba for the Qwen 2.5 VL model  
- SmolVLM team for their efficient vision-language model
- FastAPI community for the excellent web framework