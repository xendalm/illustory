import json
import os


def load_characters_data(filepath: str) -> dict:
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

def extract_characters_info(data: dict) -> list[dict]:
    """
    Извлекает и агрегирует информацию о персонажах из загруженных JSON-данных.

    Args:
        data: Словарь с данными из JSON-файла.

    Returns:
        Список словарей, где каждый словарь представляет персонажа
        и содержит его имя, связанный с ним текст и исходные предложения.
    """
    characters_data = []
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

        characters_data.append({
            "cluster_id": i,
            "representative_mention": mentions[0] if mentions else f"Персонаж {i}",
            "mentions": mentions,
            "aggregated_text_context": aggregated_text.strip(),
            "source_sentences": source_sentences
        })

    return characters_data


if __name__ == '__main__':
    example_file = "output/906.json"
    characters_data = load_characters_data(example_file)
    extracted_characters = extract_characters_info(characters_data)

    if extracted_characters:
        print(f"Найдено {len(extracted_characters)} персонажей в файле {example_file}:\n")
        for i, character in enumerate(extracted_characters):
            if i >= 3:
                print(f"... и ещё {len(extracted_characters) - i} персонажей.")
                break
            print(f"--- Персонаж: {character['representative_mention']} ---")

            print(character["aggregated_text_context"][:300] + "...")
            print("-" * (len(character["representative_mention"]) + 18) + "\n") 