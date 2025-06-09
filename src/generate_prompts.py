import os
import google.generativeai as genai
import json
import re
import time
from src.character_extractor import load_characters_data, extract_characters_info
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


def parse_llm_json_response(response_text: str) -> dict:
    """Извлекает и парсит JSON-объект из текстового ответа LLM.

    Args:
        response_text (str): Текстовый ответ от языковой модели,
                             потенциально содержащий JSON-объект.

    Returns:
        dict: Распарсенный JSON-объект в виде словаря. Возвращает None,
              если произошла ошибка парсинга.
    """
    match = re.search(r"```json\s*(\{.*?\})\s*```", response_text, re.DOTALL)
    if match:
        json_str = match.group(1)
    else:
        json_str = response_text

    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        print(f"Ошибка: Не удалось распарсить JSON из ответа модели:\n{response_text}")
        return None


def generate_visual_prompt(character_text: str, character_mentions: list[str]) -> dict:
    """Генерирует структурированное визуальное описание персонажа с помощью Gemini API.

    Args:
        character_text (str): Полный текст, из которого нужно извлечь описание.
        character_mentions (list[str]): Список упоминаний, относящихся к одному персонажу.

    Returns:
        dict: Словарь со структурированным визуальным описанием персонажа.
              Возвращает None в случае ошибки API или парсинга ответа.
    """
    model = genai.GenerativeModel('gemini-1.5-pro')
    
    # Настройки для более точной и предсказуемой генерации JSON
    generation_config = genai.GenerationConfig(
        temperature=0.2,
        max_output_tokens=2048,
        stop_sequences=["```"] # Останавливаем генерацию после JSON-блока
    )

    mentions_str = ", ".join(f'"{m}"' for m in character_mentions)

    prompt_template = f"""
    Вы — эксперт по визуализации персонажей из литературных произведений.
    Ваша задача – на основе предоставленного текста о персонаже создать подробное, детальное визуальное описание в формате JSON.
    Это описание будет использовано как промпт для модели генерации изображений (например, Stable Diffusion).
    Описание должно быть детализированным и включать следующие поля:
    - "physical_description": Общее описание внешности (возраст, пол, телосложение, рост, цвет кожи, лицо, волосы, цвет/стиль волос, цвет глаз).
    - "clothing_and_accessories": Описание одежды, стиля, цветов и аксессуаров.
    - "demeanor_and_expression": Описание типичного поведения, выражения лица.
    – "distinguishing_marks": Любые уникальные приметы (шрамы, татуировки, необычные черты).
    – "contextual_elements": Объекты или элементы окружения, часто ассоциирующиеся с персонажем.
    – "negative_prompt": Элементы, которые не должны присутствовать на изображении (например, искаженные руки, размытый фон).
    
    **Инструкции:**
    1.  Внимательно прочти весь текст для анализа.
    2.  Найди и сфокусируйся **ТОЛЬКО** на описаниях, относящихся к сущности, которую называют **любым из этих имен**: {mentions_str}.
    3.  Игнорируй описания других персонажей, даже если они упоминаются в тексте.
    4.  Сформируй JSON-объект, заполнив как можно больше полей на основе найденных фактов. Не выдумывай детали.
    5.  Если для какого-то поля нет информации, используй "Нет информации".
    Не выдумывай факты, опирайся на текст. За хороший ответ будет высокая награда.

    **Пример:**
    Входной текст: "Валентина Муттеркинда была высокой женщиной с рыжими волосами.
    Она часто носила синее платье. Её глаза были ярко-зелеными. Она всегда выглядела задумчивой."
    Выход JSON:
    ```json
    {{
      "character_name": "Валентина Муттеркинда",
      "physical_description": "Высокая женщина, стройного телосложения, с рыжими волосами и ярко-зелеными глазами.",
      "clothing_and_accessories": "Элегантное синее платье классического кроя, без заметных аксессуаров.",
      "demeanor_and_expression": "Задумчивое выражение лица, спокойная и слегка меланхоличная осанка.",
      "distinguishing_marks": "Нет",
      "contextual_elements": "Нет",
      "negative_prompt": "Размытое лицо, искаженные пропорции, лишние конечности, низкое качество."
    }}
    ```

    **Текст для анализа:**
    ---
    {character_text}
    ---
    """

    try:
        response = model.generate_content(
            prompt_template, 
            generation_config=generation_config,
            request_options={'timeout': 100}
        )
        return parse_llm_json_response(response.text)
    except Exception as e:
        print(f"Ошибка при вызове API Gemini: {e}")
        return None

def create_finetuning_dataset():
    """
    Обрабатывает все JSON-файлы в папке 'output', генерирует промпты
    и сохраняет их в файл 'finetuning_data.jsonl'.
    """
    output_dir = "output"
    if not os.path.isdir(output_dir):
        print(f"Ошибка: Директория '{output_dir}' не найдена.")
        return

    json_files = [f for f in os.listdir(output_dir) if f.endswith('.json')]
    
    print(f"Найдено {len(json_files)} JSON-файлов. Начинаем создание датасета...")

    with open(FINETUNING_DATA_FILE, "w", encoding="utf-8") as outfile:
        for filename in tqdm(json_files, desc="Обработка файлов"):
            file_path = os.path.join(output_dir, filename)
            characters_data = extract_characters_info(load_characters_data(file_path))

            if not characters_data:
                continue

            for character in tqdm(characters_data, desc=f"Персонажи в {filename}", leave=False):
                character_text = character['aggregated_text_context']
                all_mentions = character['mentions']
                
                if not all_mentions:
                    continue

                structured_prompt = generate_visual_prompt(character_text, all_mentions)

                if structured_prompt:
                    dataset_entry = {
                        "input": json.dumps({
                            "mentions": all_mentions,
                            "context": character_text
                        }, ensure_ascii=False),
                        "output": json.dumps(structured_prompt, ensure_ascii=False)
                    }
                    outfile.write(json.dumps(dataset_entry, ensure_ascii=False) + "\n")
                
                # Добавляем задержку в 1.1 секунды, чтобы не превышать лимит API
                time.sleep(2.1)
    
    print(f"\nДатасет для дообучения сохранен в файл: {FINETUNING_DATA_FILE}")


if __name__ == '__main__':
    # create_finetuning_dataset()

    #--- Блок для быстрой проверки одного файла (можно раскомментировать) ---
    example_file = "output/906.json"
    print(f"Извлечение данных из файла: {example_file}")
    extracted_data = extract_characters_info(load_characters_data(example_file))

    if extracted_data:
        for character in extracted_data:
            character_text = character['aggregated_text_context']
            character_mentions = character['mentions']
            print(f"\n--- Обработка персонажа: {character['representative_mention']} ---")
            visual_prompt = generate_visual_prompt(character_text, character_mentions)
            if visual_prompt:
                print(f"Сгенерированный промпт для '{character['representative_mention']}':")
                print(visual_prompt)
    else:
        print("Не удалось извлечь персонажей из файла.") 