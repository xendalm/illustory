import re
import string

import jsonlines
import numpy as np
import pymorphy2
from allennlp.data.tokenizers import SpacyTokenizer
from allennlp.predictors import Predictor
from nltk.corpus import stopwords
from nltk.tokenize import PunktSentenceTokenizer, word_tokenize
from sentence_transformers import SentenceTransformer, util

MAX_LEN = 400
OVERLAP = 3
SIMILARITY_THRESHOLD = 0.94

tokenizer = SpacyTokenizer()
embed_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
russian_stopwords = set(stopwords.words("russian")) | {'оно', 'такой', 'это', 'наш', 'кто', 'такое', 'тот', 'каждый',
                                                       'она', 'некоторые', 'сей', 'так', 'здесь', 'всем', 'всех',
                                                       'этого', 'вы', 'твой', 'все', 'его', 'такие', 'та', 'этот', 'мы',
                                                       'того', 'такая', 'её', 'ваш', 'те', 'какой', 'чей', 'их', 'сюда',
                                                       'они', 'всеми', 'я', 'ты', 'эти', 'там', 'мой', 'который', 'что',
                                                       'вся', 'туда', 'некоторый', 'он', 'чьё'}

morph = pymorphy2.MorphAnalyzer()
punct = set(string.punctuation)


def get_sentences_with_offsets(text: str):
    return [(text[start:end], start, end) for start, end in PunktSentenceTokenizer().span_tokenize(text)]


def split_into_chunks(sentences_with_offsets, max_tokens=MAX_LEN, overlap=OVERLAP):
    chunks, i = [], 0
    while i < len(sentences_with_offsets):
        token_count, j = 0, i
        chunk_text = ""
        chunk_start = sentences_with_offsets[i][1]

        while j < len(sentences_with_offsets):
            sent, _, _ = sentences_with_offsets[j]
            tokens = tokenizer.tokenize(sent)
            if token_count + len(tokens) > max_tokens and j > i:
                break
            token_count += len(tokens)
            chunk_text += sent + " "
            j += 1

        chunks.append((chunk_text.strip(), chunk_start))
        if j == len(sentences_with_offsets):
            break
        i = j - overlap if j - overlap > i else j
    return chunks


# COREF

def get_token_offsets(text, tokens):
    offsets, cursor = [], 0
    for token in tokens:
        start = text.find(token, cursor)
        end = start + len(token)
        offsets.append((start, end))
        cursor = end
    return offsets


def process_chunk(chunk_text, chunk_start_offset, predictor, full_text):
    result = predictor.predict(document=chunk_text)
    token_offsets = get_token_offsets(chunk_text, result["document"])

    chunk_clusters = []
    for cluster in result["clusters"]:
        mentions, global_spans = [], []
        for start, end in cluster:
            char_start, char_end = token_offsets[start][0], token_offsets[end][1]
            global_start, global_end = chunk_start_offset + char_start, chunk_start_offset + char_end
            mentions.append(full_text[global_start:global_end])
            global_spans.append((global_start, global_end))
        chunk_clusters.append({"mentions": mentions, "global_spans": global_spans})
    return chunk_clusters


# OVERLAP MERGING

def spans_overlap(span1, span2):
    return not (span1[1] <= span2[0] or span2[1] <= span1[0])


def merge_clusters_by_overlap(clusters):
    merged = []
    for cluster in sorted(clusters, key=lambda c: c['global_spans'][0]):
        for m in merged:
            if any(spans_overlap(a, b) for a in cluster['global_spans'] for b in m['global_spans']):
                m['mentions'].extend(cluster['mentions'])
                m['global_spans'].extend(cluster['global_spans'])
                break
        else:
            merged.append(cluster)
    return merged


# EMBEDDING MERGING

def clean_mention(text):
    tokens = word_tokenize(text, language="russian")
    lemmas = [
        morph.parse(token.lower())[0].normal_form
        for token in tokens
        if token.lower() not in punct and re.search(r"\w", token)
    ]
    return " ".join([lemma for lemma in lemmas if lemma not in russian_stopwords])


def get_cluster_embedding(cluster_mentions):
    cleaned = [clean_mention(m).strip() for m in cluster_mentions]
    cleaned = [c for c in cleaned if c]
    if not cleaned:
        return None
    unique_mentions, counts = np.unique(cleaned, return_counts=True)
    embeddings = embed_model.encode(unique_mentions)
    return np.average(embeddings, axis=0, weights=counts)


def merge_clusters_by_embeddings(clusters, threshold=SIMILARITY_THRESHOLD):
    merged, embeddings = [], []

    for cluster in clusters:
        cur_emb = get_cluster_embedding(cluster['mentions'])
        cur_num = len(cluster['mentions'])
        if cur_emb is None:
            continue
        for i, (emb, num) in enumerate(embeddings):
            if util.cos_sim(cur_emb, emb).item() > threshold:
                merged[i]['mentions'].extend(cluster['mentions'])
                merged[i]['global_spans'].extend(cluster['global_spans'])
                new_num = num + cur_num
                embeddings[i] = ((emb * num + cur_emb * cur_num) / new_num, new_num)
                break
        else:
            merged.append(cluster)
            embeddings.append((cur_emb, cur_num))
    return merged


def main():
    predictor = Predictor.from_path("model2.tar.gz")
    with open("input.txt", "r", encoding="utf-8") as file:
        full_text = file.read()

    sentences = get_sentences_with_offsets(full_text)
    chunks = split_into_chunks(sentences)
    print(f"Chunks count: {len(chunks)}")

    all_clusters = []
    for chunk_text, offset in chunks:
        all_clusters.extend(process_chunk(chunk_text, offset, predictor, full_text))
    print(f"Raw clusters: {len(all_clusters)}")

    overlap_merged = merge_clusters_by_overlap(all_clusters)
    print(f"After overlap merge: {len(overlap_merged)}")

    final_clusters = merge_clusters_by_embeddings(overlap_merged)
    print(f"After embedding merge: {len(final_clusters)}")

    with jsonlines.open("clusters.jsonl", "w") as writer:
        for cluster in final_clusters:
            writer.write(cluster)


if __name__ == "__main__":
    main()
