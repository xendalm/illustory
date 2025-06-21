import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen2TokenizerFast
from peft import PeftModel

class VisExtractor:
    """
    A class for extracting visual descriptions from text using a fine-tuned causal language model.
    It leverages coreference resolution to provide context-aware descriptions for entities.
    """

    def __init__(self, model_name: str = "t-tech/T-lite-it-1.0", adapters_dir: str = "t-lite-lora"):
        """
        Initializes the VisExtractor with the specified model and adapter directories.

        Args:
            model_name (str): The name or path of the base pre-trained language model.
            adapters_dir (str): The directory containing the PEFT adapters for fine-tuning.
        """
        self.model_name = model_name
        self.adapters_dir = adapters_dir
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"VisExtractor device = {self.device}")
        self._load_model_and_tokenizer()
        self.coref_pipeline = None # Initialize lazily if needed, or pass in constructor

        self.system_prompt = """Ты — эксперт по визуализации в литературных произведениях.
Твоя задача — создать максимально точное и АКТУАЛЬНОЕ визуальное описание для сущности на основе предоставленного текста.
Описание должно отражать состояние сущности на момент окончания текста, учитывая все изменения.
Это описание будет использовано как промпт для модели генерации изображений (например, Stable Diffusion).
Для персонажей делай указание на пол, возраст (если указано), чтобы генеративной модели было проще.
"""

    def _load_model_and_tokenizer(self):
        """
        Loads the base model, tokenizer, and merges the PEFT adapters.
        """
        print(f"Loading base model: {self.model_name}...")
        self.base_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        print("Base model loaded.")

        if not os.path.isdir(self.adapters_dir):
            raise FileNotFoundError(f"Adapter directory '{self.adapters_dir}' not found.")

        print(f"Loading tokenizer from adapters directory: {self.adapters_dir}...")
        # Assuming tokenizer is saved in the adapter directory for Qwen2TokenizerFast
        self.tokenizer = Qwen2TokenizerFast.from_pretrained(self.adapters_dir, trust_remote_code=True)
        print("Tokenizer loaded.")

        print(f"Loading and merging PEFT adapters from: {self.adapters_dir}...")
        self.inference_model = PeftModel.from_pretrained(self.base_model, self.adapters_dir)
        self.inference_model = self.inference_model.merge_and_unload()
        print("Adapters merged and model prepared for inference.")

    def generate_description(self, cluster_info: dict, do_sample: bool=False, temp: float=0.1) -> str:

        context = '. '.join((sentence for _, sentence, _ in cluster_info["sentences"]))
        mentions = ', '.join(cluster_info["mentions"])

        user_prompt = f"Контекст: {context} | Сущность: {', '.join(mentions)} | Создай актуальное и точное визуальное описание."
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)

        generated_ids = self.inference_model.generate(
            **model_inputs,
            max_new_tokens=256,
            temperature=temp,
            do_sample=do_sample, # Added do_sample=True for temperature to take effect
        )
        response = self.tokenizer.decode(generated_ids[0][len(model_inputs.input_ids[0]):], skip_special_tokens=True)
        return response
