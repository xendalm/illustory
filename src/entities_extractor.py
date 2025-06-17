import json
import os


def load_entities_data(filepath: str) -> dict:
    """
    Загружает JSON-данные из файла.

    Args:
        filepath (str): Путь к JSON-файлу.

    Returns:
        dict: Словарь с данными из файла или пустой словарь в случае ошибки.
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"Ошибка: Файл не найден по пути {filepath}")
        return {}
    except json.JSONDecodeError:
        print(f"Ошибка: Некорректный формат JSON в файле {filepath}")
        return {}


def extract_entities_info(data: dict) -> list[dict]:
    """
    Извлекает и агрегирует информацию о сущностях из загруженных JSON-данных.

    Args:
        data (dict): Словарь, с данными из JSON-файла.

    Returns:
        list[dict]: Список словарей, где каждый словарь представляет одну
                    сущность и содержит ее упоминания и агрегированный контекст.
    """
    entities_data = []
    clusters = data.get('clusters', [])

    for i, cluster in enumerate(clusters):
        mentions = cluster.get('mentions', [])
        sentences_info = cluster.get('sentences', [])

        # Склеиваем текст для подачи в LLM, убирая теги
        aggregated_text = " ".join([
            sentence_text.replace("<", "").replace(">", "") 
            for _, sentence_text, _ in sentences_info
        ])

        # Сохраняем исходные предложения
        source_sentences = [
            {"id": sentence_number, "text": sentence_text}
            for sentence_number, sentence_text, _ in sentences_info
        ]

        entities_data.append({
            "cluster_id": i,
            "representative_mention": mentions[0] if mentions else f"Персонаж {i}",
            "mentions": mentions,
            "aggregated_text_context": aggregated_text.strip(),
            "source_sentences": source_sentences
        })

    return entities_data


if __name__ == '__main__':
    example_file = "output/906.json"
    entities_data = load_entities_data(example_file)
    extracted_entities = extract_entities_info(entities_data)

    if extracted_entities:
        print(f"Найдено {len(extracted_entities)} сущностей в файле {example_file}:\n")
        for i, entity in enumerate(extracted_entities):
            if i >= 3:
                print(f"... и ещё {len(extracted_entities) - i} сущностей.")
                break
            print(f"--- Сущность: {entity['representative_mention']} ---")

            print(entity["aggregated_text_context"][:300] + "...")
            print("-" * (len(entity["representative_mention"]) + 18) + "\n") 