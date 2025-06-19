import os
import numpy as np
import pandas as pd
import torch
import json
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

MODEL_NAME = "ai-forever/ruBert-base"
DATASET_FILE = "labeled_sentences.jsonl"
OUTPUT_DIR = "sentence_classifier_model"
TEST_SIZE = 0.2
RANDOM_STATE = 42


def load_dataset(file_path: str) -> pd.DataFrame:
    """Загружает датасет из .jsonl файла в pandas DataFrame.

    Args:
        file_path (str): Путь к файлу датасета в формате .jsonl.

    Returns:
        pd.DataFrame: DataFrame с колонками 'sentence' и 'label'.
    """
    records = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                records.append(json.loads(line))
    except FileNotFoundError:
        print(f"Ошибка: Файл датасета не найден по пути: {file_path}")
        print("Пожалуйста, сначала создайте его с помощью sentence_labeler.py")
        exit()
    return pd.DataFrame(records)


def compute_metrics(pred) -> dict:
    """Вычисляет метрики качества (accuracy, f1, precision, recall).

    Эта функция передается в Trainer и вызывается во время оценки модели.

    Args:
        pred (EvalPrediction): Объект, содержащий предсказания модели и истинные метки.

    Returns:
        dict: Словарь с вычисленными метриками.
    """
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary"
    )
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}


class SentenceDataset(torch.utils.data.Dataset):
    """
    Класс для представления датасета в формате, совместимом с PyTorch/Hugging Face.
    """

    def __init__(self, encodings: dict, labels: list):
        """
        Args:
            encodings (dict): Словарь с токенизированными текстами
                              (выход от Hugging Face Tokenizer).
            labels (list): Список числовых меток для каждого текста.
        """
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx: int) -> dict:
        """Возвращает один элемент датасета по индексу.

        Args:
            idx (int): Индекс элемента.

        Returns:
            dict: Словарь, содержащий тензоры input_ids, attention_mask и labels.
        """
        item = {key: val[idx].detach().clone() for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self) -> int:
        """Возвращает общее количество элементов в датасете."""
        return len(self.labels)


def classify_sentences(sentences: list[str], model_path: str = "sentence_classifier_model") -> list[int]:
    """Классифицирует список предложений с использованием обученной модели.

    Args:
        sentences (list[str]): Список предложений для классификации.
        model_path (str): Путь к каталогу, содержащему обученную модель и токенизатор.

    Returns:
        list[int]: Список предсказанных меток (0 или 1) для каждого предложения.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    encodings = tokenizer(
        sentences,
        truncation=True,
        padding=True,
        max_length=512,
        return_tensors="pt")

    model.eval()

    with torch.no_grad():
        outputs = model(**encodings)

    predictions = torch.argmax(outputs.logits, dim=-1).tolist()

    return predictions

class WeightedTrainer(Trainer):
    """Кастомный Trainer с поддержкой взвешенной функции потерь для несбалансированных классов."""
    
    def __init__(self, *args, class_weights=None, **kwargs):
        """
        Args:
            *args: Стандартные аргументы для Trainer
            class_weights (torch.Tensor, optional): Веса для классов. Если None, веса будут вычислены автоматически.
            **kwargs: Дополнительные аргументы для Trainer
        """
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Вычисляет взвешенную функцию потерь.

        Args:
            model: Модель для обучения
            inputs: Входные данные
            return_outputs (bool): Возвращать ли выходы модели вместе с loss
            num_items_in_batch (int, optional): Количество элементов в батче

        Returns:
            tuple или float: (loss, outputs) если return_outputs=True, иначе только loss
        """
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        if self.class_weights is None:
            loss_fct = torch.nn.CrossEntropyLoss()
        else:
            loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights.to(model.device))

        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))

        return (loss, outputs) if return_outputs else loss

def main():
    torch.cuda.empty_cache()

    print(f"Загрузка датасета из {DATASET_FILE}...")
    df = load_dataset(DATASET_FILE)

    if df.empty:
        print("Датасет пуст. Обучение невозможно.")
        return

    texts = df["sentence"].tolist() # Теперь используем объединенный текст
    labels = df["label"].tolist()

    # Разделение на обучающую и тестовую выборки
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=labels
    )

    # Токенизация
    print(f"Загрузка токенизатора для модели {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_encodings = tokenizer(
        train_texts,
        truncation=True,
        padding=True,
        max_length=512,
        return_tensors="pt"
    )
    val_encodings = tokenizer(
        val_texts,
        truncation=True,
        padding=True,
        max_length=512,
        return_tensors="pt"
    )

    # Создание датасетов PyTorch
    train_dataset = SentenceDataset(train_encodings, train_labels)
    val_dataset = SentenceDataset(val_encodings, val_labels)

    print(f"Загрузка модели {MODEL_NAME} для классификации...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        hidden_dropout_prob=0.2,
        attention_probs_dropout_prob=0.2,
        num_labels=2)

    # Определение параметров для обучения
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        warmup_steps=1000,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        learning_rate=1e-5,
        lr_scheduler_type="cosine_with_restarts",
        fp16=True,
        optim="adamw_torch_fused",
        max_grad_norm=1.0,
        gradient_checkpointing=True
    )

    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(labels),
        y=labels
    )

    class_weights = torch.tensor(class_weights, dtype=torch.float32)

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        class_weights=class_weights
    )

    print("Начало процесса обучения...")
    trainer.train()

    # Сохранение финальной модели и токенизатора
    print(f"Обучение завершено. Сохранение модели в {OUTPUT_DIR}...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("\n--- Процесс завершен ---")


if __name__ == "__main__":
    sentences = [
        "Он поехал на машине",
        "Иван шел по широкой зеленой улице."
        "Бразильские ученые разработали алгоритм машинного обучения для автоматического обнаружения пожаров на фотографиях",
        "Валентина Муттеркинда была высокой женщиной с рыжими волосами",
        "Она часто носила синее платье. Её глаза были ярко-зелеными",
        "Она всегда шла вперед",
        "Небольшая, пыльная сцена с единственным софитом, направленным на центр. За кулисами видны старые декорации.",
        "Наброски костюмов, выполненные карандашом, с мельчайшими деталями и тенями, демонстрирующие глубокое понимание стиля и персонажей.",
        "Мужчина 60 лет, с седыми, растрёпанными волосами и блестящими глазами. Он стоит в стороне, скрестив руки на груди, его брови нахмурены.",
        "Пожелтевшие, тонкие листы бумаги, написанные размашистым, детским почерком. Некоторые из них порваны по сгибам."
    ]
    # print(classify_sentences(sentences))
    main()