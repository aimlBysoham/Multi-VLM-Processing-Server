import gc
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

class KimiModelProcessor:
    def __init__(self, model_path="moonshotai/Kimi-VL-A3B-Thinking"):
        self.model_path = model_path
        self.model = None
        self.processor = None


    def load_model(self):
        """Loads the model and processor from the provided model path."""
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
        )
        self.processor = AutoProcessor.from_pretrained(
            self.model_path, 
            trust_remote_code=True
        )

    def unload_model(self):
        """
        Unloads the model and processor from GPU, clears cache, and frees Python memory.
        """
        # Move model to CPU (if not already)
        if self.model is not None:
            try:
                self.model.to("cpu")
            except Exception:
                pass

        # Delete model and processor references
        del self.model
        del self.processor

        # Clear CUDA memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Force Python GC
        gc.collect()

        # Reset attributes
        self.model = None
        self.processor = None


    def process(self, prompt, image_path=None):
        """
        Processes the image and text prompt, and returns the model's output as a text string.
        
        Parameters:
            image_path (str): The file path to the image.
            prompt (str): The text prompt describing what to ask about the image.
            
        Returns:
            str: The generated textual output from the model.
        """

        if image_path is not None:
            # Open the image from the given path.
            image = Image.open(image_path)
            
            # Create the chat message for processing.
            messages = [
                {
                    "role": "user", 
                    "content": [
                        {"type": "image", "image": image_path},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
            # Apply chat template to convert messages into the text input format.
            text = self.processor.apply_chat_template(
                messages, 
                add_generation_prompt=True, 
                return_tensors="pt"
            )
            
            # Prepare inputs from image and text.
            inputs = self.processor(
                images=image, 
                text=text, 
                return_tensors="pt", 
                padding=True, 
                truncation=True
            ).to(self.model.device)
            
            # Generate response from the model.
            generated_ids = self.model.generate(**inputs, max_new_tokens=8192)
            
            # Trim the input part from the generated output.
            generated_ids_trimmed = [
                out_ids[len(in_ids):] 
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            
            # Decode the generated token IDs to get the textual output.
            response = self.processor.batch_decode(
                generated_ids_trimmed, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=False
            )[0]

        else:
            # Create the chat message for processing.
            messages = [
                {
                    "role": "user", 
                    "content": [
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
            # Apply chat template to convert messages into the text input format.
            text = self.processor.apply_chat_template(
                messages, 
                add_generation_prompt=True, 
                return_tensors="pt"
            )
            
            # Prepare inputs from image and text.
            inputs = self.processor(
                text=text, 
                return_tensors="pt", 
                padding=True, 
                truncation=True
            ).to(self.model.device)
            
            # Generate response from the model.
            generated_ids = self.model.generate(**inputs, max_new_tokens=8192)
            
            # Trim the input part from the generated output.
            generated_ids_trimmed = [
                out_ids[len(in_ids):] 
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            
            # Decode the generated token IDs to get the textual output.
            response = self.processor.batch_decode(
                generated_ids_trimmed, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=False
            )[0]
        
        return response


# if __name__ == "__main__":

#     model_processor = KimiModelProcessor()

#     frames_main_dir = 'Testing_track_Ids/images'
#     output_main_dir = 'Testing_track_Ids/output_texts_kimi_vl_a3b'
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

#             prompt_text = "Describe me the person in the image."

#             # Process the image and prompt to get the description
#             description = model_processor.process(image_path=image_path, prompt=prompt_text)

#             print(description)
            
#             output_path = os.path.join(output_dir, f"{os.path.splitext(image_name)[0]}.txt")
#             with open(output_path, "w") as f:
#                 f.write(description)
