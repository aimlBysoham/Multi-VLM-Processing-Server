import os
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
    image_path: str
    prompt_text: str

# Define the processing endpoint
@app.post("/process-image")
async def process_image(payload: ImageRequestPayload):
    try:
        image_path = payload.image_path
        prompt_text = payload.prompt_text

        # Ensure the image exists
        if not os.path.exists(image_path):
            raise HTTPException(status_code=400, detail=f"Image file not found at {image_path}")

        # Process the image and prompt to get the description with Kimi
        kimi_processor.load_model()
        kimi_description = kimi_processor.process(prompt=prompt_text, image_path=image_path)
        kimi_processor.unload_model()

        # Process the image and prompt to get the description with Qwen
        qwen_processor.load_model()
        qwen_description = qwen_processor.process(prompt=prompt_text, image_path=image_path)
        qwen_processor.unload_model()

        # Process the image and prompt to get the description with SmolVLM
        smolvlm_processor.load_model()
        smolvlm_description = smolvlm_processor.process(prompt=prompt_text, image_path=image_path)
        smolvlm_processor.unload_model()

        # Return the results
        return JSONResponse(content={
            "kimi_description": kimi_description,
            "qwen_description": qwen_description,
            "smolvlm_description": smolvlm_description
        })
    except Exception as e:
        return HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Run the FastAPI app on port 7090
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7090)
