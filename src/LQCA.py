import os
import json
import time
from types import SimpleNamespace
from collections import defaultdict, Counter
import argparse # Импортируем argparse

import requests
import pymorphy2
import transformers
import tensorflow as tf
from datasets import load_dataset, DatasetDict
from natasha import Segmenter, NewsEmbedding, NewsSyntaxParser, NewsMorphTagger, Doc
import json

# Предполагается, что LongCorpipe установлен и доступен
from LongCorpipe import corpipe24 as cor
from LongCorpipe.clusterer import merge_clusters

# --- Глобальные конфигурации и инициализация ---

# Настройки для UDPipe
UDPIPE_PORT = 8001

# Настройки для Corpipe
CORPIPE_ARGS = {
    "encoder"           : "google/mt5-large",
    "segment"           : 2560,
    "right"             : 50,
    "zeros_per_parent"  : 2,
    "batch_size"        : 18,
    "threads"           : 32,
    "load"              : "LongCorpipe/corpipe24-corefud1.2-240906/",
    "depth"             : 5,
}
CORPIPE_ARGS = SimpleNamespace(**CORPIPE_ARGS)
gpu = tf.config.list_physical_devices('GPU')[0]
tf.config.experimental.set_memory_growth(gpu, True)

# Инициализация Corpipe токенизатора и модели
tf.config.threading.set_inter_op_parallelism_threads(CORPIPE_ARGS.threads)
corpipe_tokenizer = transformers.AutoTokenizer.from_pretrained(CORPIPE_ARGS.encoder)
corpipe_tokenizer.add_special_tokens({"additional_special_tokens": [cor.Dataset.TOKEN_EMPTY] +
                                    [cor.Dataset.TOKEN_TREEBANK.format(i) for i in range(21)] +
                                    ([cor.Dataset.TOKEN_CLS] if corpipe_tokenizer.cls_token_id is None else [])})

with open(os.path.join(CORPIPE_ARGS.load, "tags.txt"), mode="r") as tags_file:
    corpipe_tags = [line.rstrip("\r\n") for line in tags_file]
with open(os.path.join(CORPIPE_ARGS.load, "zdeprels.txt"), mode="r") as zdeprels_file:
    corpipe_zdeprels = [line.rstrip("\r\n") for line in zdeprels_file]

corpipe_tags_map = {tag: i for i, tag in enumerate(corpipe_tags)}
corpipe_zdeprels_map = {zdeprel: i for i, zdeprel in enumerate(corpipe_zdeprels)}

# Инициализация модели Corpipe один раз
corpipe_model = cor.Model(corpipe_tokenizer, corpipe_tags, corpipe_zdeprels, CORPIPE_ARGS)

# Инициализация Natasha для морфологического анализа и синтаксического разбора
morph = pymorphy2.MorphAnalyzer()
segmenter = Segmenter()
embedding = NewsEmbedding()
syntax_parser = NewsSyntaxParser(embedding)
morph_tagger = NewsMorphTagger(embedding)

# Настройки для Natasha и грамматических правил
BAD_TAGS = {'PRON', 'DET', 'PUNCT'}
GRAMMEME_MAP = {
    'Nom': 'nomn',
    'Gen': 'gent',
    'Dat': 'datv',
    'Acc': 'accs',
    'Ins': 'ablt',
    'Loc': 'loct',
    'Sing': 'sing',
    'Plur': 'plur',
    'Pos': 'Poss',
}
ALLOWED_KEYS = {'Case', 'Number'}

# --- Вспомогательные функции ---

def process_text_local_udpipe(text, port=UDPIPE_PORT, output_format='conllu'):
    """
    Обрабатывает текст с помощью локального сервера UDPipe.
    """
    url = f'http://localhost:{port}/process'
    params = {
        'tokenizer': 'ranges',
        'tagger': '',
        'parser': '',
        'data': text,
        'output': output_format,
    }
    try:
        response = requests.post(url, data=params)
        response.raise_for_status()
        result_json = response.json()
        if 'result' in result_json:
            return result_json['result']
        else:
            raise Exception(f"Ключ 'result' не найден в ответе от сервера: {result_json}")
    except json.JSONDecodeError:
        raise Exception(f"Сервер вернул невалидный JSON. Текст ответа: {response.text}")
    except requests.exceptions.RequestException as e:
        raise Exception(f"Ошибка при подключении к локальному серверу UDPipe: {e}")

def norm_mention(mention_info):
    """
    Нормализует упоминание, приводя токены к нормальной форме и отфильтровывая стоп-теги.
    """
    tokens = [
        morph.parse(token.lower())[0].normal_form
        for token, _, pos in mention_info
        if pos not in BAD_TAGS
    ]
    tags = [pos for _, _, pos in mention_info]

    if not any(tag in {"PROPN", "NOUN"} for tag in tags):
        return None
    return tuple(tokens)

def get_mention_grammemes(doc: Doc, mention):
    """
    Извлекает граммемы для упоминания из объекта Natasha Doc.
    """
    token_id_to_idx_map = {token.id: i for i, token in enumerate(doc.tokens)}
    head_token = None
    mention_tokens = doc.tokens[mention.begin : mention.end + 1]
    # Ищем токен внутри упоминания, чей родитель (head) находится вне его.
    for token in mention_tokens:
        head_id_str = token.head_id
        head_idx = token_id_to_idx_map.get(head_id_str)
        if head_idx is None or not (mention.begin <= head_idx <= mention.end):
            head_token = token
            break

    if not head_token and mention_tokens:
        # Ищем последнее существительное/имя собственное
        for token in reversed(mention_tokens):
            if token.pos in ('NOUN', 'PROPN'):
                head_token = token
                break
        # Если и так не нашли, берем просто последний токен
        if not head_token:
            head_token = mention_tokens[-1]

    # Извлекаем граммемы из найденного головного токена.
    if head_token:
        feats = head_token.feats
        grammemes = set()
        if feats:
            for key, value in feats.items():
                if key in ALLOWED_KEYS and value in GRAMMEME_MAP:
                    grammemes.add(GRAMMEME_MAP[value])
        return grammemes
        
    return set()

def inflect_phrase(phrase_words: list[str], target_grammemes: set) -> str:
    """
    Склоняет слова в фразе в соответствии с целевыми граммемами.
    """
    inflected_words = []
    for word in phrase_words:
        parses = morph.parse(word)
        # Выбираем лучший разбор, который имеет наибольшее пересечение граммем с целевыми
        best_parse = max(parses, key=lambda p: len(p.tag.grammemes & target_grammemes))
        inflected = best_parse.inflect(target_grammemes)
        if inflected:
            # Сохраняем регистр первого слова
            if word and word[0].isupper():
                inflected_words.append(inflected.word.capitalize())
            else:
                inflected_words.append(inflected.word)
        else:
            inflected_words.append(word)
    return " ".join(inflected_words)

def should_replace(mention_info, normalized_mention, cluster_repr):
    """
    Определяет, следует ли заменять упоминание на репрезентативное выражение кластера.
    """
    if normalized_mention == cluster_repr:
        return False

    POS_tags = [pos for _, _, pos in mention_info]
    if all(t in BAD_TAGS for t in POS_tags):
        return False
    
    # Не заменять, если все токены - имена собственные и репрезентативное выражение короче
    if all(t == 'PROPN' for t in POS_tags) and len(cluster_repr) <= len(normalized_mention or ()):
        return False
    
    return True

# --- Основной пайплайн LQCA ---

def lqca_pipeline(dataset_name, dataset_subset, context_field, output_dir):
    """
    Выполняет полный пайплайн LQCA:
    1. Загрузка датасета.
    2. Преобразование текста в CoNLL-U формат.
    3. Разрешение кореференций с помощью Corpipe.
    4. Замена упоминаний на репрезентативные выражения кластеров.
    5. Сохранение обработанного датасета.
    """
    print(f"Загрузка датасета: {dataset_name}, подмносество: {dataset_subset}")
    dataset = load_dataset(dataset_name, dataset_subset)["test"]
    os.makedirs(output_dir, exist_ok=True)

    def process_item(item, item_idx):
        original_context = item[context_field]
        print(f"\nОбработка элемента {item_idx + 1}...")
        replacements_count = 0

        # 1. Преобразование в CoNLL-U
        try:
            print("Преобразование контекста в CoNLL-U...")
            conllu_text = process_text_local_udpipe(original_context)
            # Временно сохраняем в файл для Corpipe
            temp_conllu_path = f"temp_doc_{item_idx}.conllu"
            with open(temp_conllu_path, "w", encoding="utf-8") as f:
                f.write(conllu_text)
            print("CoNLL-U успешно создан.")
        except Exception as e:
            print(f"Ошибка при преобразовании в CoNLL-U: {e}. Возвращаем исходный элемент.")
            if os.path.exists(temp_conllu_path):
                os.remove(temp_conllu_path)
            return {**item, 'original_context': original_context, 'enriched_context': original_context, 'replacements_made': 0}

        # 2. Разрешение кореференций с помощью Corpipe
        try:
            print("Запуск Corpipe для разрешения кореференций...")
            # Использование класса Dataset из Corpipe для чтения CoNLL-U
            books = cor.Dataset(temp_conllu_path, corpipe_tokenizer, 0)
            generator = books.pipeline(corpipe_tags_map, corpipe_zdeprels_map, False, CORPIPE_ARGS).ragged_batch(CORPIPE_ARGS.batch_size).prefetch(tf.data.AUTOTUNE)
            predicts = corpipe_model.predict(books, generator)

            corpiped_mentions = [[] for _ in range(len(books.docs))]
            cluster_add = 0
            for doc_i, predict in enumerate(predicts):
                local_results = [[] for _ in range(len(predict))]
                clusters = merge_clusters(predict)
                for cluster_id, cluster in enumerate(clusters):
                    for mention in cluster:
                        mention.cluster = cluster_id + cluster_add
                        local_results[mention.sent_id].append(mention)
                cluster_add += len(clusters)
                for i in range(len(local_results)):
                    local_results[i] = sorted(local_results[i], key=lambda x: (x.begin, -getattr(x, 'is_zero', False), -x.end))
                corpiped_mentions[doc_i] = local_results
            print("Кореференции разрешены.")

        except Exception as e:
            print(f"Ошибка при разрешении кореференций с Corpipe: {e}. Возвращаем исходный элемент.")
            if os.path.exists(temp_conllu_path):
                os.remove(temp_conllu_path)
            return {**item, 'original_context': original_context, 'enriched_context': original_context, 'replacements_made': 0}
        finally:
            # Удаляем временный CoNLL-U файл
            if os.path.exists(temp_conllu_path):
                os.remove(temp_conllu_path)

        if not corpiped_mentions:
            print("Corpipe не вернул упоминаний для данного документа. Возвращаем исходный контекст.")
            return {**item, 'original_context': original_context, 'enriched_context': original_context, 'replacements_made': 0}
        
        # Обработка только первого документа, так как dataset_item - это один контекст
        doc_mentions = corpiped_mentions[0] # Берем первый (и единственный) документ

        # Получаем все предложения из books.docs_flu
        all_sentences_tokens = []
        for sent_info in books.docs_flu[0]: # Берем первый (и единственный) документ
            words = [token_info[0] for token_info in sent_info]
            all_sentences_tokens.append(words)

        # 3. Замена упоминаний на репрезентативные выражения кластеров
        print("Замена упоминаний на репрезентативные выражения...")
        
        # Шаг 3.1: Построение кластеров и их репрезентаций
        clusters_map = defaultdict(list)
        for sent_mentions in doc_mentions:
            for mention in sent_mentions:
                clusters_map[mention.cluster].append(mention)

        clusters_repr = {}
        for cluster_id, cluster in clusters_map.items():
            cluster_candidates = []
            for mention in cluster:
                normalized_mention = norm_mention(mention.info) # Используем mention.info
                if normalized_mention:
                    cluster_candidates.append(normalized_mention)
            
            if cluster_candidates:
                clusters_repr[cluster_id] = Counter(cluster_candidates).most_common(1)[0][0]
        
        # Шаг 3.2: Замена в предложениях
        enriched_sentences_tokens = [list(sent) for sent in all_sentences_tokens] # Копируем, чтобы не изменять оригинал

        # ключ - ID кластера, значение - set с нормализованными формами, которые уже встретились.
        seen_normalized_mentions_in_cluster = defaultdict(set)

        # итерируемся по предложениям в ХРОНОЛОГИЧЕСКОМ порядке
        for sent_id, sent_mentions in enumerate(doc_mentions):
            if not sent_mentions:
                continue

            sentence_text = " ".join(all_sentences_tokens[sent_id])
            doc_natasha = Doc(sentence_text)
            doc_natasha.segment(segmenter)
            doc_natasha.tag_morph(morph_tagger)
            doc_natasha.parse_syntax(syntax_parser)

            replacements_to_make = []
            # Принимаем решения о замене
            for mention in sent_mentions:
                cluster_id = mention.cluster
                if cluster_id not in clusters_repr: # у кластера мог не найтись "достойный" представитель
                    continue

                normalized_mention = norm_mention(mention.info)
                if not normalized_mention:
                    continue

                # Проверяем, видели ли мы уже это нормализованное упоминание в данном кластере
                if normalized_mention in seen_normalized_mentions_in_cluster[cluster_id]:
                    cluster_representation = clusters_repr[cluster_id]
                    
                    if should_replace(mention.info, normalized_mention, cluster_representation):
                        target_grammemes = get_mention_grammemes(doc_natasha, mention)
                        if not target_grammemes:
                            continue
                        
                        inflected_repr = inflect_phrase(list(cluster_representation), target_grammemes)
                        # Добавляем в список запланированных замен
                        replacements_to_make.append((mention, inflected_repr))
                else:
                    # Если это первое вхождение, то мы его НЕ заменяем и добавляем в увиденные для данного кластера.
                    seen_normalized_mentions_in_cluster[cluster_id].add(normalized_mention)
            
            # Увеличиваем счетчик на количество замен в этом предложении
            replacements_count += len(replacements_to_make)

            current_sentence = enriched_sentences_tokens[sent_id]
            # Сортируем упоминания в обратном порядке по begin, чтобы замены не влияли на индексы последующих упоминаний в том же предложении
            for mention, replacement_text in sorted(replacements_to_make, key=lambda x: x[0].begin, reverse=True):
                enriched_sentences_tokens[sent_id] = (
                    current_sentence[:mention.begin] +
                    [replacement_text] +
                    current_sentence[mention.end + 1:]
                )

        enriched_context = " ".join([" ".join(s) for s in enriched_sentences_tokens])
        dct = {**item, 'original_context': original_context, 'enriched_context': enriched_context, 'replacements_made': replacements_count}
        with open(f"{output_dir}/current.jsonl", "a", encoding="utf-8") as f:
            json.dump(dct, f, ensure_ascii=False)
            f.write('\n')
        return dct

    # Применение функции process_item ко всему датасету
    # Используем map для эффективной обработки

    processed_dataset = dataset.map(lambda item, idx: process_item(item, idx), with_indices=True)

    # 4. Сохранение обработанного датасета
    processed_dataset.save_to_disk(output_dir)
    print(f"\nОбработанный датасет сохранен в: {output_dir}")
    
    return processed_dataset

# --- Запуск пайплайна с CLI аргументами ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LQCA Pipeline for Coreference Resolution and Context Enrichment.")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="ai-forever/LIBRA",
        help="Название датасета для загрузки (например, 'ai-forever/LIBRA')."
    )
    parser.add_argument(
        "--dataset_subset",
        type=str,
        default="long_context_multiq",
        help="Подмножество датасета для использования (например, 'ru_babilong_qa5')."
    )
    parser.add_argument(
        "--context_field",
        type=str,
        default="context",
        help="Название поля в датасете, содержащего текст для обработки кореференций."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="processed_dataset",
        help="Путь для сохранения обработанного датасета."
    )
    parser.add_argument(
        "--udpipe_port",
        type=int,
        default=8001,
        help="Порт, на котором работает локальный сервер UDPipe."
    )

    args = parser.parse_args()

    # Обновляем глобальные переменные на основе CLI аргументов
    UDPIPE_PORT = args.udpipe_port

    processed_dataset = lqca_pipeline(
        dataset_name=args.dataset_name,
        dataset_subset=args.dataset_subset,
        context_field=args.context_field,
        output_dir=args.output_dir
    )

    print("\nПайплайн LQCA завершен.")