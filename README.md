# Multi-VLM Processing Server

A FastAPI-based application that processes images using multiple Vision Language Models (VLMs) to generate comprehensive descriptions. This project integrates three different VLM models: Kimi VL, Qwen 2.5 VL, and SmolVLM to provide diverse perspectives on image analysis.

## Features

- **Multi-Model Processing**: Utilizes three different VLM models for comprehensive image analysis
- **REST API**: FastAPI-based web service for easy integration
- **Concurrent Processing**: Processes images with all three models and returns combined results
- **Fashion & Lifestyle Focus**: Optimized for product descriptions in fashion and lifestyle contexts
- **Error Handling**: Robust error handling with appropriate HTTP status codes

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
        └── temp.jpg                     # Sample image for testing
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

### Running the FastAPI Server

Start the web server on port 7090:

```bash
python src/server/rest_api/app.py
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
    "image_path": "path/to/your/image.jpg",
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
       "image_path": "src/utils/temp.jpg",
       "prompt_text": "Describe me the product in the image."
     }'
```

### Running the Standalone Script

For testing individual models or batch processing:

```bash
python main.py
```

This will process the sample image (`src/utils/temp.jpg`) with all three models and display the results.

## Model Processing Flow

1. **Model Loading**: Each VLM processor loads its respective model
2. **Image Processing**: The image is processed with the provided prompt
3. **Description Generation**: Each model generates a description based on the image and prompt
4. **Model Unloading**: Models are unloaded to free up memory
5. **Response Compilation**: All descriptions are combined and returned

## Error Handling

The API includes comprehensive error handling:

- **400 Bad Request**: When the specified image file is not found
- **500 Internal Server Error**: For any processing errors during model execution

## Performance Considerations

- Models are loaded and unloaded for each request to manage memory usage
- Processing time varies depending on image size and model complexity
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