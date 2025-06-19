import os
import json
import time
import re
import random
import google.generativeai as genai
from dotenv import load_dotenv
from tqdm import tqdm

# Загружаем переменные окружения из файла .env
load_dotenv()

# --- Конфигурация API ---
try:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("API-ключ GEMINI не найден. Установите переменную окружения GEMINI_API_KEY.")
    genai.configure(api_key=api_key)
except (AttributeError, ValueError) as e:
    print(f"Ошибка конфигурации API: {e}")
    exit()

# Модель для быстрой классификации
CLASSIFICATION_MODEL = genai.GenerativeModel('gemini-2.0-flash')
OUTPUT_FILE = "labeled_sentences.jsonl"
LABELED_DATA_FILE = "labeled_sentences.jsonl"
SOURCE_FILE = "sentences_for_labeling.txt"


def label_sentence_batch(sentences_batch: list[str]) -> list[int]:
    """
    Классифицирует пачку предложений с помощью Gemma API.
    Возвращает список словарей ({"label": 1/0, "reason": "..."}) или пустой список в случае ошибки.
    """
    sentences_json = json.dumps(sentences_batch, ensure_ascii=False, indent=2)

    prompt = f"""
    Ты — эксперт по анализу текста, специализирующийся на выделении визуальных описаний.
    Твоя задача — классифицировать КАЖДОЕ предложение из предоставленного JSON-массива. Для каждого предложения ты должен определить, содержит ли оно визуальное описание.

    **Что является визуальным описанием:**
    * **Персонаж**: общее описание внешности (возраст, пол, телосложение, рост, цвет кожи, лицо, волосы, цвет/стиль волос, цвет глаз), одежды, стиля, аксессуаров,
    типичного поведения, характера, выражения лица, любых уникальных примет.
    * **Локация/Объект/Место:**: описание архитектуры, интерьера, мебели, природы, пейзажа, освещения, цветов, атмосферы и других деталей, создающих образ места.

    **Что НЕ является визуальным описанием:**
        * Простое упоминание имени персонажа или названия места/объекта без уточняющих деталей ("Он пошел в магазин", "Они встретились в Париже", "Стол стоял посередине").
        * Описание абстрактных понятий, эмоций или мыслей, не имеющих прямого визуального или сенсорного воплощения ("Он был добрым человеком", "Она чувствовала грусть", "Идея показалась гениальной").
        * Прямые действия, которые не описывают внешний вид или окружение ("Он быстро бежал", "Она подумала о завтрашнем дне", "Договор был подписан").
        * Описания, которые являются метафорами, не имеющими прямого визуального эквивалента.

    **Примеры для ОДНОГО предложения:**
    - "Он накинул свою потертую кожаную куртку." -> {{"label": 1, "reason": "Присутствует описание конкретного предмета одежды ('потертая кожаная куртка')."}}
    - "Они жили на Невском проспекте." -> {{"label": 0, "reason": "Указание на место без каких-либо визуальных деталей."}}
    - "Старинный замок возвышался на скале, его шпили пронзали свинцовые тучи." -> {{"label": 1, "reason": "Дается яркое визуальное описание замка и окружающей его атмосферы."}}
    - "Я думаю, нам следует уходить, – сказал он." -> {{"label": 0, "reason": "Предложение передает прямую речь и намерение, а не визуальное описание."}}
    - "Её глаза сияли, как два изумруда на бледном лице." -> {{"label": 1, "reason": "Используется метафора для описания цвета глаз и лица, что создает визуальный образ."}}

    **Предложения для анализа (JSON-массив):**
    ---
    {sentences_json}
    ---

    **Твой ответ должен быть ТОЛЬКО в формате JSON-массива из {len(sentences_batch)} объектов. Каждый объект должен содержать два поля: "label" (число 1 или 0) и "reason" (строка с краткой аргументацией на русском).**
    Пример формата ответа:
    [
        {{"label": 1, "reason": "Аргументация для первого предложения."}},
        {{"label": 0, "reason": "Аргументация для второго предложения."}},
        {{"label": 1, "reason": "Аргументация для третьего предложения."}}
    ]
    **Не добавляй никаких других пояснений, только JSON-массив.**
    """

    try:
        response = CLASSIFICATION_MODEL.generate_content(prompt)

        match = re.search(r'```json\s*([\s\S]*?)\s*```|(\[[\s\S]*\])', response.text)
        if not match:
            print(f"Ошибка: Не удалось найти JSON в ответе модели: {response.text}")
            return []

        json_str = match.group(1) or match.group(2)
        answers = json.loads(json_str)

        if len(answers) != len(sentences_batch):
            print(f"Ошибка: Количество ответов ({len(answers)}) не совпадает с количеством предложений ({len(sentences_batch)}).")
            return []

        # Проверка формата ответа
        for ans in answers:
            if not isinstance(ans, dict) or "label" not in ans or "reason" not in ans:
                 print(f"Ошибка: Неверный формат элемента в ответе: {ans}")
                 return []

        return answers
    except json.JSONDecodeError:
        print(f"Ошибка декодирования JSON из ответа: {response.text}")
        return []
    except Exception as e:
        print(f"Ошибка при обработке батча: {e}")
        return []


def label_sentences_from_file():
    """
    Читает предложения, обрабатывает их пачками по 5 и записывает результаты.
    """
    try:
        with open(SOURCE_FILE, 'r', encoding='utf-8') as f:
            sentences_to_label = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"Ошибка: Исходный файл не найден: '{SOURCE_FILE}'")
        return

    print(f"Найдено {len(sentences_to_label)} предложений для разметки.")

    batch_size = 20
    with open(LABELED_DATA_FILE, "a", encoding="utf-8") as outfile:
        with tqdm(total=len(sentences_to_label), desc=f"Разметка предложений (Gemma API, батчи по {batch_size})") as pbar:
            for i in range(0, len(sentences_to_label), batch_size):
                batch = sentences_to_label[i:i + batch_size]
                if not batch:
                    continue

                results = label_sentence_batch(batch)
                
                if len(results) == len(batch):
                    for sentence, result in zip(batch, results):
                        record = {"sentence": sentence, "label": result.get("label"), "reason": result.get("reason")}
                        outfile.write(json.dumps(record, ensure_ascii=False) + "\n")
                else:
                    print(f"Пропуск батча из-за ошибки обработки. Индексы предложений: {i}-{i+len(batch)-1}")

                pbar.update(len(batch))
                time.sleep(1) # Небольшая задержка между батчами

    print(f"\nРазметка завершена. Данные добавлены в {LABELED_DATA_FILE}")


if __name__ == '__main__':
    label_sentences_from_file()