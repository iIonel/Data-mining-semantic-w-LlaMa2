import torch
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
)


class TokenizerLoader:
    @staticmethod
    def build(name):
        tok = AutoTokenizer.from_pretrained(name)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        tok.padding_side = "right"
        return tok


class QuantizationConfig:
    @staticmethod
    def nf4(qlora):
        compute_dtype = getattr(torch, qlora.get("bnb_4bit_compute_dtype", "float16"))
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=qlora.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_use_double_quant=qlora.get("bnb_4bit_use_double_quant", True),
            bnb_4bit_compute_dtype=compute_dtype,
        )


class ModelBuilder:
    def __init__(self, config, num_labels, smoke=False):
        self.config = config
        self.num_labels = num_labels
        self.smoke = smoke
        self.use_quant = (not smoke) and torch.cuda.is_available()
        self.name = config["model"]["smoke_name"] if smoke else config["model"]["name"]

    def build(self):
        tok = TokenizerLoader.build(self.name)
        model = self._load_base()
        model.config.pad_token_id = tok.pad_token_id
        if self.use_quant:
            model = self._attach_lora(model)
            model.print_trainable_parameters()
        return model, tok

    def _load_base(self):
        if self.use_quant:
            model = AutoModelForSequenceClassification.from_pretrained(
                self.name,
                num_labels=self.num_labels,
                quantization_config=QuantizationConfig.nf4(self.config["qlora"]),
                device_map="auto",
            )
            return prepare_model_for_kbit_training(model)
        return AutoModelForSequenceClassification.from_pretrained(
            self.name, num_labels=self.num_labels,
        )

    def _attach_lora(self, model):
        qlora = self.config["qlora"]
        lora = LoraConfig(
            r=qlora["r"],
            lora_alpha=qlora["lora_alpha"],
            lora_dropout=qlora["lora_dropout"],
            target_modules=qlora["target_modules"],
            bias=qlora.get("bias", "none"),
            task_type=TaskType.SEQ_CLS,
            modules_to_save=qlora.get("modules_to_save", ["score"]),
        )
        return get_peft_model(model, lora)
