# Импорт необходимых библиотек
from sentence_transformers import SentenceTransformer, util
import rapidfuzz

model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

def match_keys(gold_keys, pred_keys):
    # Fuzzy matching ключей
    matches = {}
    for gk in gold_keys:
        best_match = None
        best_score = 0
        for pk in pred_keys:
            score = rapidfuzz.fuzz.ratio(gk, pk)
            if score > best_score:
                best_score = score
                best_match = pk
        # if best_score > 80:  # порог можно подобрать
        matches[gk] = best_match
    return matches

def evaluate_key_matching(gold, pred):
    gold_keys = list(gold.keys())
    pred_keys = list(pred.keys())
    if not pred_keys:
        return 0.0
    matches = match_keys(gold_keys, pred_keys)
    # print(f"Key matches: {matches}")
    scores = []
    for gk, pk in matches.items():
        emb_gold = model.encode(gold[gk], convert_to_tensor=True)
        emb_pred = model.encode(pred[pk], convert_to_tensor=True)
        sim = util.pytorch_cos_sim(emb_gold, emb_pred).item()
        scores.append(sim)
    return sum(scores) / len(scores) if scores else 0.0

def evaluate_description_matching(gold, pred):
    matches = {}
    scores = []
    for gk, gv in gold.items():
        best_match = None
        best_score = 0
        for pk, pv in pred.items():
            if not isinstance(pv, str):
                print(f"Skipping non-string prediction value for key '{pk}': {pv}")
                pv = ""
            emb_gold = model.encode(gv, convert_to_tensor=True)
            emb_pred = model.encode(pv, convert_to_tensor=True)
            score = util.pytorch_cos_sim(emb_gold, emb_pred).item()
            if score > best_score:
                best_score = score
                best_match = pk
        matches[gk] = best_match
        scores.append(best_score)
    # print(f"Description matches: {matches}")
    return sum(scores)/len(scores) if scores else 0.0


def calculate_scores(gold, pred):
    key_matching_scores = [evaluate_key_matching(gold[i], pred[i]) for i in range(len(gold))]
    description_matching_scores = [evaluate_description_matching(gold[i], pred[i]) for i in range(len(gold))]
    return key_matching_scores, description_matching_scores
