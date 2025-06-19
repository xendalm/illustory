import os
import google.generativeai as genai
import json
import re
import time
from src.entities_extractor import load_entities_data, extract_entities_info
from sentence_classifier import classify_sentences
from tqdm import tqdm
from dotenv import load_dotenv

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

# Файл для сохранения данных для дообучения
FINETUNING_DATA_FILE = "finetuning_data.jsonl"
GEMINI_MODEL = "gemini-2.0-flash"
CLASSIFIER_MODEL_PATH = "sentence_classifier_model"


def generate_visual_prompt(entity_text: str, entity_mentions: list[str]) -> dict | None:
    """Генерирует структурированное визуальное описание сущности с помощью Gemini API.

    Args:
        entity_text (str): Полный текст, из которого нужно извлечь актуальное описание.
        entity_mentions (list[str]): Список упоминаний, относящихся к одной сущности.

    Returns:
        str: Актуальное визуальное описание сущности.
             Возвращает None в случае ошибки API или парсинга ответа.
    """
    model = genai.GenerativeModel(GEMINI_MODEL)

    generation_config = genai.GenerationConfig(
        temperature=0.2,
    )

    mentions_str = ", ".join(f'"{m}"' for m in entity_mentions)

    prompt_template = f"""
    Вы — эксперт по визуализации в литературных произведениях.
    Ваша задача — создать максимально точное, детальное и АКТУАЛЬНОЕ визуальное описание для сущности на основе предоставленного текста.
    Описание должно отражать состояние сущности на момент окончания текста, учитывая все изменения.
    В тексте все упоминания данной сущности выделены угловыми скобками <...>.
    Это описание будет использовано как промпт для модели генерации изображений (например, Stable Diffusion).

    **Инструкции:**
    1.  Внимательно прочти весь текст для анализа.
    2.  Сконцентрируйся ТОЛЬКО на сущности, которую называют {mentions_str}.
    3.  Создай детальное визуальное описание, включающее:
       - Для персонажей: возраст, внешность, одежда, выражение лица, поза, эмоции, уникальные приметы
       - Для локаций: атмосфера, освещение, настроение, ключевые объекты, цвета
       - Для предметов: внешний вид, состояние, детали
    4. Не выдумывай детали, опирайся только на текст.
    5. Описание должно быть в виде одного связного текста без угловых скобок.
    6. Описание должно быть актуальным.
    7. Помни, что полученное описание будет использоваться как промпт для генерации иллюстрации.

    Например, если в начале текста персонаж был в шляпе, а в конце снял её, в описании шляпы быть не должно.
    Если он получил шрам, шрам должен быть в описании.
    Если в локации что-то поменялось (например, зажглись свечи), отрази это.
    Аналогично для объектов/предметов.

    Делай указание на пол, возраст (если указано), чтобы генеративной модели было проще.

    **Текст для анализа:**
    ---
    {entity_text}
    ---
    """

    try:
        response = model.generate_content(
            prompt_template, 
            generation_config=generation_config,
            request_options={'timeout': 100}
        )
        return response.text.strip()
    except Exception as e:
        print(f"Ошибка при вызове API Gemini: {e}")
        return None

def create_finetuning_dataset():
    """
    Обрабатывает все JSON-файлы в папке 'output', генерирует визуальные промпты
    и сохраняет их в файл 'finetuning_data.jsonl'.
    """
    output_dir = "output"
    if not os.path.isdir(output_dir):
        print(f"Ошибка: Директория '{output_dir}' не найдена.")
        return

    json_files = [f for f in os.listdir("output") if f.endswith('.json')]

    print(f"Найдено {len(json_files)} JSON-файлов. Начинаем создание датасета...")

    with open(FINETUNING_DATA_FILE, "a", encoding="utf-8") as outfile:
        for filename in tqdm(json_files, desc="Обработка файлов"):
            file_path = os.path.join(output_dir, filename)
            extracted_data = extract_entities_info(load_entities_data(file_path))

            if not extracted_data:
                continue

            for entity in tqdm(extracted_data, desc=f"Сущности в {filename}", leave=False):
                source_sentences = entity['source_sentences']
                all_mentions = entity['mentions']

                if not all_mentions:
                    continue

                # --- Шаг 1: Фильтрация предложений ---
                original_sentences = [s['text'] for s in source_sentences]
                clean_sentences = [s['text'].replace("<", "").replace(">", "") for s in source_sentences]
                try:
                    # Используем обученную модель
                    predictions = classify_sentences(clean_sentences, CLASSIFIER_MODEL_PATH)
                    filtered_sentences = [
                        original_sentences[i] for i, label in enumerate(predictions) if label == 1
                    ]
                except Exception as e:
                    continue

                if not filtered_sentences:
                    continue

                filtered_context = " ".join(filtered_sentences)

                # --- Шаг 2: Генерация промпта на основе отфильтрованного контекста ---
                visual_description = generate_visual_prompt(filtered_context, all_mentions)

                time.sleep(2.5)

                if visual_description:
                    dataset_entry = {
                        "mentions": all_mentions,
                        "context": filtered_context,
                        "output": visual_description
                    }
                    outfile.write(json.dumps(dataset_entry, ensure_ascii=False) + "\n")

            print(f"Файл {filename} обработан успешно.")

    print(f"\nПайплайн завершен. Датасет для дообучения сохранен в файл: {FINETUNING_DATA_FILE}.")


if __name__ == '__main__':
    create_finetuning_dataset()
    # json_files = [f for f in os.listdir("output") if f.endswith('.json')]
    # print(json_files)