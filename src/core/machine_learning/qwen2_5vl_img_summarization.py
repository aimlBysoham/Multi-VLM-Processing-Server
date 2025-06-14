import gc
import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor


class QwenModelProcessor:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        self.model = None
        self.processor = None
        

    def load_model(self):
        """Load the processor and model."""
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_name, torch_dtype="auto", device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(self.model_name)

    def unload_model(self):
        """Unload the processor and model to free resources."""

        if self.model is not None:
            try:
                self.model.to("cpu")
            except Exception:
                pass

            del self.model
            self.model = None

        if self.processor is not None:
            del self.processor
            self.processor = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        gc.collect()

    def process(self, prompt: str, image_path: str) -> str:
        """Process the image and prompt to generate a text description."""
        if not self.model or not self.processor:
            raise ValueError("Model not loaded. Please call load_model() first.")
        
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_path,
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)

        generated_ids = self.model.generate(**inputs, max_new_tokens=1024)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        return output_text[0]



# if __name__ == "__main__":
#     # Initialize the processor and model
#     vision_processor = QwenModelProcessor()

#     frames_main_dir = '/home/soham/Projects/Interaction_VLM/vd_to_img_summary/Testing_track_Ids/images'
#     output_main_dir = '/home/soham/Projects/Interaction_VLM/vd_to_img_summary/Testing_track_Ids/output_texts_qwen2_5vl'
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

#             print(f"Processed {image_name}: {description}")
            
            # output_path = os.path.join(output_dir, f"{os.path.splitext(image_name)[0]}.txt")
            # with open(output_path, "w") as f:
            #     f.write(description)
