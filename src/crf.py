import os
import sys
import nltk
import argparse
import subprocess
import jsonlines
from nltk.tokenize import sent_tokenize, wordpunct_tokenize
from sentence_transformers import SentenceTransformer, util

MAX_LEN = 400
TEMP_DIR = "temp_coref_chunks"
os.makedirs(TEMP_DIR, exist_ok=True)

embed_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

def split_into_chunks(text, max_tokens=MAX_LEN):
    sentences = sent_tokenize(text, language='russian')
    chunks = []
    current_chunk = []
    token_count = 0

    for sent in sentences:
        tokens = wordpunct_tokenize(sent)
        if token_count + len(tokens) > max_tokens and current_chunk:
            chunks.append(current_chunk)
            current_chunk = [sent]
            token_count = len(tokens)
        else:
            current_chunk.append(sent)
            token_count += len(tokens)

    if current_chunk:
        chunks.append(current_chunk)

    return [' '.join(chunk) for chunk in chunks]

def convert_to_conll(chunk_text, doc_id="book", part=0):
    lines = [f"#begin document ({doc_id}); part {part}"]
    token_index = 0
    for sent in sent_tokenize(chunk_text, language='russian'):
        lines.append("")
        for token in wordpunct_tokenize(sent):
            line = f"{doc_id}\t0\t{token_index}\t{token}\t_\t_\t_\t_\t_\t_\t-"
            lines.append(line)
            token_index += 1
    lines.append("#end document")
    return "\n".join(lines)

def run_coref_model(model_path, input_path, output_path):
    subprocess.run([
        "allennlp", "evaluate", model_path, input_path,
        "--include-package", "allennlp_models",
        "--predictions-output-file", output_path
    ], capture_output=True)

def extract_mentions(json_path):
    mentions = []
    with jsonlines.open(json_path) as reader:
        for obj in reader.iter():
            mentions.append(obj["clusters"][0])
    return mentions

def mention_to_text(span, tokens):
    return ' '.join(tokens[span[0]:span[1]+1])

def get_cluster_embedding(cluster, tokens):
    texts = [mention_to_text(span, tokens) for span in cluster]
    embeddings = embed_model.encode(texts, convert_to_tensor=True)
    return embeddings.mean(dim=0)

def merge_clusters_by_embeddings(all_mentions, all_tokens, threshold=0.9):
    merged_clusters = []
    embeddings = []

    for chunk_idx, (chunk_mentions, chunk_tokens) in enumerate(zip(all_mentions, all_tokens)):
        for cluster in chunk_mentions[0]:
            cluster_emb = get_cluster_embedding(cluster, chunk_tokens)
            found_match = False

            for i, existing_emb in enumerate(embeddings):
                sim = util.cos_sim(cluster_emb, existing_emb).item()
                if sim > threshold:
                    merged_clusters[i].append((chunk_idx, cluster))
                    embeddings[i] = (embeddings[i] + cluster_emb) / 2
                    found_match = True
                    break

            if not found_match:
                merged_clusters.append([(chunk_idx, cluster)])
                embeddings.append(cluster_emb)

    return merged_clusters


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="Path to the coref model", required=True)
    args = parser.parse_args()

    with open("input.txt", "r", encoding="utf-8") as file:
        full_text = file.read()

    chunks = split_into_chunks(full_text, MAX_LEN)
    print(len(chunks))

    all_mentions = []
    all_chunk_tokens = []

    for i, chunk_text in enumerate(chunks):
        conll_text = convert_to_conll(chunk_text, doc_id="doc", part=i)
        temp_in = os.path.join(TEMP_DIR, f"chunk_{i}.conll")
        temp_out = os.path.join(TEMP_DIR, f"chunk_{i}_out.jsonl")
        with open(temp_in, 'w', encoding='utf-8') as f:
            f.write(conll_text)

        run_coref_model(args.model, temp_in, temp_out)

        mentions = extract_mentions(temp_out)
        tokens = wordpunct_tokenize(chunk_text)
        all_mentions.append(mentions)
        all_chunk_tokens.append(tokens)

    print(all_mentions)

    merged = merge_clusters_by_embeddings(all_mentions, all_chunk_tokens)

    token_offsets = []
    current_offset = 0
    for tokens in all_chunk_tokens:
        token_offsets.append(current_offset)
        current_offset += len(tokens)


    for i, cluster_group in enumerate(merged):
        mentions_with_spans = []
        for chunk_idx, cluster in cluster_group:
            tokens = all_chunk_tokens[chunk_idx]
            offset = token_offsets[chunk_idx]
            for span in cluster:
                text = mention_to_text(span, tokens)
                global_span = (span[0] + offset, span[1] + offset)
                mentions_with_spans.append((text, span, global_span, chunk_idx))
        print(f"\nCluster M{i+1}:")
        for text, local_span, global_span, chunk_idx in mentions_with_spans:
            print(f"  - {text} (span={local_span}, global_span={global_span}, chunk={chunk_idx})")



if __name__ == "__main__":
    main()