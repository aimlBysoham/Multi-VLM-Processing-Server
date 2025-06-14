import gc
import torch
from PIL import Image
from transformers.image_utils import load_image
from transformers import AutoProcessor, AutoModelForVision2Seq


class SmolVlmModelProcessor:
    def __init__(self, model_name: str = "HuggingFaceTB/SmolVLM-500M-Instruct"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        self.processor = None
        self.model = None
        

    def load_model(self):
        """Load the processor and model."""
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = (
            AutoModelForVision2Seq.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16,
            )
            .to(self.device)
        )


    def unload_model(self):
        """Unload the processor and model to free CPU & GPU memory."""

        if getattr(self, "model", None) is not None:
            try:
                self.model.to("cpu")
            except Exception:
                pass
            del self.model
            self.model = None


        if getattr(self, "processor", None) is not None:
            del self.processor
            self.processor = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        gc.collect()


    def process(self, prompt: str, image_path: str) -> str:
        """Process the image and prompt to generate a text description."""
        image = load_image(image_path)

        if not self.processor or not self.model:
            raise ValueError("Model not loaded. Please call load_model() first.")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt}
                ]
            },
        ]

        chat_prompt = self.processor.apply_chat_template(
            messages, add_generation_prompt=True
        )
        inputs = self.processor(
            text=chat_prompt, images=[image], return_tensors="pt"
        ).to(self.device)

        generated_ids = self.model.generate(**inputs, max_new_tokens=1024)
        generated_texts = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )

        return generated_texts[0]




# if __name__ == "__main__":

#     vision_processor = SmolVlmModelProcessor()

#     frames_main_dir = 'Testing_track_Ids/images'
#     output_main_dir = 'Testing_track_Ids/output_texts_smolvlm500m'
#     os.makedirs(output_main_dir, exist_ok=True)

#     for folder_name in os.listdir(frames_main_dir):

#         folder_path = os.path.join(frames_main_dir, folder_name)
#         output_dir = os.path.join(output_main_dir, folder_name)
#         os.makedirs(output_dir, exist_ok=True)

#         if not os.path.isdir(folder_path):
#             continue

#         for image_name in os.listdir(folder_path):

#             image_path = os.path.join(folder_path, image_name)

#             if not image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
#                 continue

#             prompt = "Describe me the person in the image."

#             # Process the image and prompt to get the description
#             description = vision_processor.process(image_path, prompt)

#             answer = description.split("Assistant:")[-1].strip()
            
#             output_path = os.path.join(output_dir, f"{os.path.splitext(image_name)[0]}.txt")
#             with open(output_path, "w") as f:
#                 f.write(answer)
