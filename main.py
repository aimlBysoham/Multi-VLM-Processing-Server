import os
import time

from src.core.machine_learning.qwen2_5vl_img_summarization import QwenModelProcessor
from src.core.machine_learning.smolvlm_img_summarization import SmolVlmModelProcessor
from src.core.machine_learning.kimi_vl_A3B_instruct_summarization import KimiModelProcessor


if __name__ == "__main__":

    kimi_processor = KimiModelProcessor()
    qwen_processor = QwenModelProcessor()
    smolvlm_processor = SmolVlmModelProcessor()

    image_path = 'src/utils/temp.jpg'
    image_name = os.path.basename(image_path)
        
    # Define the prompt text
    prompt_text = f"Describe me the product in the image."

    # Process the image and prompt to get the description with kimi
    kimi_processor.load_model()

    kimi_description = kimi_processor.process(prompt=prompt_text, image_path=image_path)

    print(f"\n\nKimi Description: {kimi_description}")

    kimi_processor.unload_model()

    # Process the image and prompt to get the description with qwen
    qwen_processor.load_model()

    qwen_description = qwen_processor.process(prompt=prompt_text, image_path=image_path)

    print(f"\n\nQwen Description: {qwen_description}")

    qwen_processor.unload_model()

    # Process the image and prompt to get the description with smolvlm
    smolvlm_processor.load_model()

    smolvlm_description = smolvlm_processor.process(prompt=prompt_text, image_path=image_path)

    print(f"\n\nSmolVLM Description: {smolvlm_description}")

    smolvlm_processor.unload_model()
    
