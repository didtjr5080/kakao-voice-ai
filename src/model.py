"""경량 챗봇 모델 (LoRA + 4bit 양자화)"""

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModel
)
from datasets import Dataset
from typing import List, Tuple, Optional
import os

class LightweightChatbot:
    """경량 한국어 챗봇 모델 (VRAM 1.5GB 이하)"""
    
    def __init__(
        self, 
        model_name: str = "skt/kogpt2-base-v2",
        load_in_4bit: bool = True,
        device: str = "auto"
    ):
        self.model_name = model_name
        self.load_in_4bit = load_in_4bit
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        
        self.tokenizer = None
        self.model = None
    
    def load_model(self, model_path: Optional[str] = None, use_lora: bool = False):
        """모델 로드 (4bit 양자화 적용)"""
        
        print(f"Loading model: {model_path or self.model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path or self.model_name
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        if self.load_in_4bit and self.device == "cuda":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path or self.model_name,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path or self.model_name,
                trust_remote_code=True
            )
            if self.device == "cuda":
                self.model = self.model.to(self.device)
        
        if use_lora and model_path:
            try:
                self.model = PeftModel.from_pretrained(self.model, model_path)
                print("LoRA adapter loaded")
            except:
                print("No LoRA adapter found")
        
        self.model.eval()
        print(f"Model loaded (Device: {self.device})")
        
        if self.device == "cuda":
            print(f"VRAM: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    
    def train(
        self,
        training_pairs: List[Tuple[str, str]],
        output_dir: str = "./models/kakao-chatbot",
        epochs: int = 3,
        batch_size: int = 1,
        learning_rate: float = 2e-4,
        use_lora: bool = True,
        lora_r: int = 8,
        lora_alpha: int = 16
    ):
        """모델 학습 (LoRA 파인튜닝)"""
        
        print("\n" + "="*60)
        print(">>> Model Training Started")
        print("="*60)
        
        if self.model is None:
            self.load_model()
        
        special_tokens = {
            'additional_special_tokens': ['<|user|>', '<|bot|>', '<|end|>']
        }
        self.tokenizer.add_special_tokens(special_tokens)
        self.model.resize_token_embeddings(len(self.tokenizer))
        
        if use_lora:
            print(f"LoRA config (r={lora_r}, alpha={lora_alpha})")
            
            if self.load_in_4bit:
                self.model = prepare_model_for_kbit_training(self.model)
            
            peft_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=["c_attn", "c_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM"
            )
            
            self.model = get_peft_model(self.model, peft_config)
            self.model.print_trainable_parameters()
        
        print(f"\nPreparing dataset ({len(training_pairs)} samples)")
        
        training_texts = [
            f"<|user|>{inp}<|bot|>{out}<|end|>"
            for inp, out in training_pairs
        ]
        
        print("\nTraining data samples:")
        for i, text in enumerate(training_texts[:3], 1):
            print(f"  {i}. {text[:100]}...")
        
        dataset = Dataset.from_dict({'text': training_texts})
        
        def tokenize_function(examples):
            return self.tokenizer(
                examples['text'],
                truncation=True,
                max_length=256,
                padding='max_length'
            )
        
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=['text']
        )
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=4,
            learning_rate=learning_rate,
            fp16=self.device == "cuda",
            logging_steps=10,
            save_steps=100,
            save_total_limit=2,
            warmup_steps=50,
            gradient_checkpointing=True,
            optim="adamw_torch",
            report_to="none"
        )
        
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=data_collator
        )
        
        print("\nTraining started...")
        trainer.train()
        
        print(f"\nSaving model to: {output_dir}")
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        print("\n>>> Training Complete!")
        return output_dir
    
    def generate_response(
        self,
        user_input: str,
        max_length: int = 100,
        temperature: float = 0.8,
        top_p: float = 0.9,
        top_k: int = 50
    ) -> str:
        """응답 생성"""
        
        if self.model is None or self.tokenizer is None:
            return "Model not loaded"
        
        prompt = f"<|user|>{user_input}<|bot|>"
        
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=256
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.encode('<|end|>')[0] if '<|end|>' in self.tokenizer.get_vocab() else self.tokenizer.eos_token_id,
                repetition_penalty=1.2
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
        
        if '<|bot|>' in response:
            response = response.split('<|bot|>')[-1]
            response = response.split('<|end|>')[0].strip()
        
        return response
