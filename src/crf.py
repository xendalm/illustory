import os
import nltk
import numpy as np
from nltk.corpus import stopwords
import re
import pymorphy2
import string
import nltk
import argparse
import subprocess
import jsonlines
import tempfile
from nltk.tokenize import sent_tokenize, wordpunct_tokenize
from sentence_transformers import SentenceTransformer, util

MAX_LEN = 400
OVERLAP = 2

embed_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
russian_stopwords = set(stopwords.words("russian"))
morph = pymorphy2.MorphAnalyzer()
punct = set(string.punctuation)

def tokenize_sentences(text):
    sentences = sent_tokenize(text, language='russian')
    sent_tokens = [wordpunct_tokenize(sent) for sent in sentences]
    return sentences, sent_tokens

def split_into_chunks(sentences, sent_tokens, max_tokens=MAX_LEN, overlap=OVERLAP):
    chunks = []
    i = 0
    while i < len(sentences):
        chunk_tokens = []
        chunk_sents = []
        j = i
        while j < len(sentences):
            tokens = sent_tokens[j]
            if len(chunk_tokens) + len(tokens) > max_tokens and chunk_sents:
                break
            chunk_sents.append(sentences[j])
            chunk_tokens.extend(tokens)
            j += 1
        start_token_idx = sum(len(t) for t in sent_tokens[:i])  # точный глобальный старт
        chunks.append((chunk_sents, chunk_tokens, start_token_idx))
        if j == len(sentences):
            break
        i = j - overlap if j - overlap > i else j
    return chunks

def convert_to_conll(chunk_sents, doc_id="doc", part=0):
    lines = [f"#begin document ({doc_id}); part {part}"]
    token_index = 0
    for sent in chunk_sents:
        lines.append("")
        for token in wordpunct_tokenize(sent):
            line = f"{doc_id}\t0\t{token_index}\t{token}\t_\t_\t_\t_\t_\t_\t-"
            lines.append(line)
            token_index += 1
    lines.append("#end document")
    return "\n".join(lines)

def run_coref_model(model_path, input_path, output_path):
    result = subprocess.run([
        "allennlp", "evaluate", model_path, input_path,
        "--include-package", "allennlp_models",
        "--predictions-output-file", output_path
    ], capture_output=True, text=True)

    if result.returncode != 0:
        print("Error:", result.stderr)

def run_coref_on_chunk(conll_data, model_path):
    with tempfile.NamedTemporaryFile(mode='w+', suffix=".conll", delete=False, encoding='utf-8') as temp_in, \
         tempfile.NamedTemporaryFile(mode='r', suffix=".jsonl", delete=False, encoding='utf-8') as temp_out:
        temp_in.write(conll_data)
        temp_in.flush()

        run_coref_model(model_path, temp_in.name, temp_out.name)

        temp_out.seek(0)
        mentions = extract_mentions(temp_out.name)

    os.remove(temp_in.name)
    os.remove(temp_out.name)

    return mentions

def extract_mentions(json_path):
    mentions = []
    with jsonlines.open(json_path) as reader:
        for obj in reader.iter():
            mentions.extend(obj["clusters"][0])
    return mentions

def mention_to_text(span, tokens):
    return ' '.join(tokens[span[0]:span[1]+1])

def spans_overlap(span1, span2):
    return not (span1[1] < span2[0] or span2[1] < span1[0])

def merge_clusters_by_overlap(clusters):
    merged = []
    for cluster in sorted(clusters, key=lambda c: c['global_spans'][0]):
        added = False
        for m in merged:
            if any(spans_overlap(a, b) for a in cluster['global_spans'] for b in m['global_spans']):
                m['mentions'].extend(cluster['mentions'])
                m['global_spans'].extend(cluster['global_spans'])
                added = True
                break
        if not added:
            merged.append(cluster)
    return merged


def clean_mention(text):
    tokens = nltk.word_tokenize(text, language="russian")
    lemmas = []
    for token in tokens:
        token_lower = token.lower()
        if token_lower in punct or not re.search(r"\w", token_lower):
            continue
        lemma = morph.parse(token_lower)[0].normal_form
        if lemma in russian_stopwords:
            continue
        lemmas.append(lemma)
    return " ".join(lemmas)


def get_cluster_embedding(cluster_mentions):
    mentions = [mention[0] for mention in cluster_mentions]

    cleaned = []
    for m in mentions:
        cleaned_m = clean_mention(m).strip()
        if len(cleaned_m) > 0:
            cleaned.append(cleaned_m)

    if not cleaned:
        return None

    unique_mentions, counts = np.unique(cleaned, return_counts=True)
    embeddings = embed_model.encode(unique_mentions)
    weighted_avg = np.average(embeddings, axis=0, weights=counts)

    return weighted_avg

def merge_clusters_by_embeddings(clusters, threshold=0.93):
    merged = []
    embeddings = []

    for cluster in clusters:
        emb = get_cluster_embedding(cluster['mentions'])
        if emb is None:
            continue
        found = False
        for i, e in enumerate(embeddings):
            if util.cos_sim(emb, e).item() > threshold:
                merged[i]['mentions'].extend(cluster['mentions'])
                merged[i]['global_spans'].extend(cluster['global_spans'])
                embeddings[i] = (embeddings[i] + emb) / 2
                found = True
                break
        if not found:
            merged.append(cluster)
            embeddings.append(emb)
    return merged


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    args = parser.parse_args()

    with open("input.txt", "r", encoding="utf-8") as file:
        full_text = file.read()

    sentences, sent_tokens = tokenize_sentences(full_text)
    chunks = split_into_chunks(sentences, sent_tokens)
    print(len(chunks))

    # all_tokens = [token for sent in sent_tokens for token in sent]
    all_clusters = []
    for i, (chunk_sents, chunk_tokens, global_start_idx) in enumerate(chunks):
        conll = convert_to_conll(chunk_sents, part=i)
        mentions = run_coref_on_chunk(conll, args.model)

        for cluster in mentions:
            cluster_data = {
                "mentions": [],
                "global_spans": [],
            }
            for span in cluster:
                global_span = (span[0] + global_start_idx, span[1] + global_start_idx)
                mention_text = mention_to_text(span, chunk_tokens)
                cluster_data["mentions"].append((mention_text, global_span))
                cluster_data["global_spans"].append(global_span)
            all_clusters.append(cluster_data)

    print(len(all_clusters))

    clusters_overlap_merged = merge_clusters_by_overlap(all_clusters)
    print(len(clusters_overlap_merged))

    final_clusters = merge_clusters_by_embeddings(clusters_overlap_merged)
    print(len(final_clusters))

    for i, cluster in enumerate(final_clusters):
        print(f"\nCluster M{i+1}:")
        for mention_text, global_span in cluster['mentions']:
            print(f"  - {mention_text} (global span={global_span})")

if __name__ == "__main__":
    main()