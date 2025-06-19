import os
import json
import random
from tqdm import tqdm

# --- Константы ---
OUTPUT_DIR = "output"
LABELED_DATA_FILE = "labeled_sentences_gemini.jsonl" # Файл с результатами
RESULT_FILE = "sentences_for_labeling.txt"  # Файл с новой выборкой
SAMPLE_SIZE = 70000
MIN_WORDS = 2


def get_already_labeled_sentences() -> set:
    """Читает уже размеченные предложения из файла с результатами.

    Returns:
        set: Множество уже размеченных предложений (строк).
    """
    if not os.path.exists(LABELED_DATA_FILE):
        return set()
    
    labeled_sentences = set()
    with open(LABELED_DATA_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                labeled_sentences.add(json.loads(line)['sentence'])
            except (json.JSONDecodeError, KeyError):
                continue
    return labeled_sentences


def create_sample_for_labeling():
    """
    Собирает все уникальные предложения, исключает уже размеченные,
    фильтрует и создает случайную выборку для последующей разметки.
    """
    # Получаем список уже размеченных предложений
    already_labeled = get_already_labeled_sentences()
    if already_labeled:
        print(f"Найдено {len(already_labeled)} уже размеченных предложений. Они будут исключены.")

    # Собираем все предложения из JSON-файлов
    unique_sentences = set()

    if not os.path.isdir(OUTPUT_DIR):
        print(f"Ошибка: Директория '{OUTPUT_DIR}' не найдена.")
        return

    json_files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith('.json')]
    print(f"Найдено {len(json_files)} JSON файлов. Собираю все предложения...")

    for filename in tqdm(json_files, desc="Чтение файлов"):
        file_path = os.path.join(OUTPUT_DIR, filename)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if 'clusters' in data:
                for cluster in data['clusters']:
                    for _, sentence_text, _ in cluster.get('sentences', []):
                        sentence_text = sentence_text.replace("<", "").replace(">", "")
                        unique_sentences.add(sentence_text.strip())
        except Exception as e:
            print(f"\nПроблема с файлом {filename}: {e}")
    
    print(f"Всего найдено {len(unique_sentences)} уникальных предложений в исходных файлах.")

    # Исключаем уже размеченные предложения
    fresh_sentences = unique_sentences - already_labeled
    print(f"Осталось {len(fresh_sentences)} 'свежих' предложений для выборки.")

    # Фильтрация по длине
    print(f"Фильтрация предложений (минимум {MIN_WORDS} слов)...")
    filtered_sentences = [
        s for s in fresh_sentences if len(s.split()) >= MIN_WORDS
    ]
    print(f"Осталось {len(filtered_sentences)} предложений после фильтрации.")

    # Создание случайной выборки
    if not filtered_sentences:
        print("Нет 'свежих' предложений для создания выборки. Вы разметили всё!")
        return
        
    if len(filtered_sentences) < SAMPLE_SIZE:
        print(f"Внимание: Предложений после фильтрации ({len(filtered_sentences)}) меньше, чем требуемый размер выборки.")
        final_sample = list(filtered_sentences) # Преобразуем set в list для random.sample
    else:
        print(f"Создаю случайную выборку размером {SAMPLE_SIZE}...")
        final_sample = random.sample(list(filtered_sentences), SAMPLE_SIZE)

    # Сохранение в файл
    print(f"Сохраняю выборку в файл '{RESULT_FILE}'...")
    with open(RESULT_FILE, 'w', encoding='utf-8') as f:
        for sentence in final_sample:
            f.write(sentence + '\n')

    print("-" * 20)
    print(f"Файл '{RESULT_FILE}' создан и содержит {len(final_sample)} предложений для разметки.")
    print("-" * 20)


if __name__ == "__main__":
    create_sample_for_labeling() 