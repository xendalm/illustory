import os
import json
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

# --- Конфигурация ---
# Модель для дообучения
MODEL_NAME = "t-tech/T-lite-it-1.0"
# Путь к датасету для дообучения
DATASET_PATH = "finetuning_data.jsonl"
# Директория для сохранения дообученных адаптеров
OUTPUT_DIR = "t-lite-lora"


def create_prompt(sample):
    """
    Форматирует один пример из датасета в формат чата.
    """
    system_prompt = """Ты — эксперт по визуализации в литературных произведениях.
    Твоя задача — создать максимально точное и АКТУАЛЬНОЕ визуальное описание для сущности на основе предоставленного текста.
    Описание должно отражать состояние сущности на момент окончания текста, учитывая все изменения.
    Это описание будет использовано как промпт для модели генерации изображений (например, Stable Diffusion).
    Для персонажей делай указание на пол, возраст (если указано), чтобы генеративной модели было проще.
    """

    mentions_str = ", ".join(sample['mentions'])
    user_prompt = f"Контекст: {sample['context']} | Сущность: {mentions_str} | Создай актуальное и точное визуальное описание."

    # Структура "сообщений" для модели
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
        {"role": "model", "content": sample['output']} 
    ]

    return {"text": tokenizer.apply_chat_template(messages, tokenize=False)}


def main():
    torch.cuda.empty_cache()

    print(f"Загрузка базовой модели: {MODEL_NAME}")

    # Настройка для 4-битной квантизации
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # Загрузка модели с квантизацией
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    # Загрузка токенизатора
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Модель и токенизатор загружены.")

    # --- Настройка LoRA (Peft) ---
    # Подготовка модели к k-битному обучению
    model = prepare_model_for_kbit_training(model)

    # Конфигурация LoRA
    # r - ранг (чем больше, тем больше обучаемых параметров)
    # target_modules - какие слои в модели мы будем "адаптировать"
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Применяем LoRA к модели
    model = get_peft_model(model, peft_config)
    print("LoRA адаптеры применены к модели. Обучаемые параметры:")
    model.print_trainable_parameters()

    print(f"Загрузка и форматирование данных из {DATASET_PATH}")
    dataset = load_dataset("json", data_files=DATASET_PATH, split="train")

    # Форматируем датасет под формат чата
    formatted_dataset = dataset.map(create_prompt)
    print("Пример отформатированных данных (для трейна):")
    print(formatted_dataset[0]['text'])

    # --- Запуск обучения ---
    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        num_train_epochs=3,
        max_seq_length=8192,
        lr_scheduler_type="cosine",
        fp16=True,
        logging_steps=10,
        save_strategy="epoch",
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=formatted_dataset,
        peft_config=peft_config,
        formatting_func=lambda x: x["text"],
        processing_class=tokenizer,
        args=training_args,
    )

    print("\nНачинаем дообучение модели с LoRA...")
    trainer.train()
    print("Дообучение завершено!")

    # --- Запуск адаптеров ---
    print(f"Сохранение LoRA адаптеров в директорию: {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Адаптеры и токенизатор успешно сохранены.")


def test_lora_inference():
    """
    Функция для проверки, как использовать дообученные LoRA адаптеры.
    """
    print("\n--- Тестирование модели с дообученными LoRA адаптерами ---")
    if not os.path.isdir(OUTPUT_DIR):
        print(f"Директория с адаптерами '{OUTPUT_DIR}' не найдена.")
        return

    # Загружаем базовую моедль
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    # Загружаем токенизатор
    tokenizer = AutoTokenizer.from_pretrained(OUTPUT_DIR, trust_remote_code=True)

    # Создаем PEFT модель, объединяя базовую модель и наши дообученные адаптеры
    inference_model = PeftModel.from_pretrained(base_model, OUTPUT_DIR)

    # Объединяем веса для ускорения инференса
    inference_model = inference_model.merge_and_unload()

    # Пример для теста
    test_context = """На ежегодном собрании торговой гильдии, где воздух был пропитан запахом старых книг
    и свежезаваренного кофе, <один молодой англичанин> выделялся среди степенных купцов.
    Это был <М. Б. Кларк>, на вид лет тридцати пяти, с копной рыжеватых волос, которые всегда казались
    немного растрепанными, и пронзительными голубыми глазами, внимательно изучающими каждого собеседника.
    Его строгий, но хорошо сшитый твидовый костюм и небрежно повязанный галстук выделяли его на фоне
    традиционных темных сюртуков других <коммерсантов>. Когда он говорил, его губы растягивались в легкой,
    почти незаметной усмешке, а на щеках проступали едва заметные ямочки. Именно <Кларк> предложил новую,
    смелую стратегию для развития заморской торговли, чем сразу привлек внимание присутствующих."""
    test_mentions = ["кларк", "компаньон", "один молодой англичанин", "м. б. кларк", "коммерсант"]
    
    system_prompt = system_prompt = """Ты — эксперт по визуализации в литературных произведениях.
    Твоя задача — создать максимально точное и АКТУАЛЬНОЕ визуальное описание для сущности на основе предоставленного текста.
    Описание должно отражать состояние сущности на момент окончания текста, учитывая все изменения.
    Это описание будет использовано как промпт для модели генерации изображений (например, Stable Diffusion).
    Для персонажей делай указание на пол, возраст (если указано), чтобы генеративной модели было проще.
    """
    user_prompt = f"Контекст: {test_context} | Сущность: {', '.join(test_mentions)} | Создай актуальное и точное визуальное описание."

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    # Применяем шаблон чата
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(inference_model.device)

    # Генерация ответа
    generated_ids = inference_model.generate(
        **model_inputs,
        max_new_tokens=256,
        temperature=0.1,
    )
    # Убираем входные токены из результата
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):]
    
    response = tokenizer.decode(output_ids, skip_special_tokens=True)

    print("\n--- Результат генерации ---")
    print(response)


if __name__ == "__main__":
    # main()
    test_lora_inference()