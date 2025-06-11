import os
import base64
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from src.core.machine_learning.kimi_vl_A3B_instruct_summarization import KimiModelProcessor
from src.core.machine_learning.qwen2_5vl_img_summarization import QwenModelProcessor
from src.core.machine_learning.smolvlm_img_summarization import SmolVlmModelProcessor

app = FastAPI()

# Initialize processors
kimi_processor = KimiModelProcessor()
qwen_processor = QwenModelProcessor()
smolvlm_processor = SmolVlmModelProcessor()

# Pydantic model for the request payload
class ImageRequestPayload(BaseModel):
    image_base64: str  # Base64 encoded image data
    prompt_text: str

def save_base64_image(base64_data: str, file_path: str) -> None:
    """
    Save base64 encoded image data to a file.
    
    Args:
        base64_data: Base64 encoded image string
        file_path: Path where the image should be saved
    """
    try:
        # Remove data URL prefix if present (e.g., "data:image/jpeg;base64,")
        if base64_data.startswith('data:'):
            base64_data = base64_data.split(',', 1)[1]
        
        # Decode base64 data
        image_data = base64.b64decode(base64_data)
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Write image data to file
        with open(file_path, 'wb') as f:
            f.write(image_data)
            
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to decode and save base64 image: {str(e)}")

# Define the processing endpoint
@app.post("/process-image")
async def process_image(payload: ImageRequestPayload):
    try:
        base64_image = payload.image_base64
        prompt_text = payload.prompt_text
        
        # Define the temporary image path
        temp_image_path = 'src/utils/temp.jpg'
        
        # Save base64 image to temporary file
        save_base64_image(base64_image, temp_image_path)
        
        # Verify the image was saved successfully
        if not os.path.exists(temp_image_path):
            raise HTTPException(status_code=500, detail="Failed to save temporary image file")

        # Process the image and prompt to get the description with Kimi
        kimi_processor.load_model()
        kimi_description = kimi_processor.process(prompt=prompt_text, image_path=temp_image_path)
        kimi_processor.unload_model()

        # Process the image and prompt to get the description with Qwen
        qwen_processor.load_model()
        qwen_description = qwen_processor.process(prompt=prompt_text, image_path=temp_image_path)
        qwen_processor.unload_model()

        # Process the image and prompt to get the description with SmolVLM
        smolvlm_processor.load_model()
        smolvlm_description = smolvlm_processor.process(prompt=prompt_text, image_path=temp_image_path)
        smolvlm_processor.unload_model()

        # Clean up temporary file (optional)
        try:
            os.remove(temp_image_path)
        except:
            pass  # Ignore cleanup errors

        # Return the results
        return JSONResponse(content={
            "kimi_description": kimi_description,
            "qwen_description": qwen_description,
            "smolvlm_description": smolvlm_description
        })
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        # Clean up temporary file on error
        try:
            if 'temp_image_path' in locals():
                os.remove(temp_image_path)
        except:
            pass
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Run the FastAPI app on port 7090
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7090)