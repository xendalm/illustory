import argparse
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Qwen2TokenizerFast
from peft import PeftModel
import torch
from src.CorefPipeline import CorefPipeline
from src.VisExtractor import VisExtractor
from eval.eval import calculate_scores
import json

parser = argparse.ArgumentParser()
parser.add_argument("--udpipe_port", default=8001, type=int, help="Localhost udpipe port")
args = parser.parse_args()

current_dir = os.getcwd()
if not current_dir.endswith('illustory'):
    raise RuntimeError(f"Скрипт должен запускаться из корневой папки illustory. Текущая папка: {current_dir}")

descriptions_dataset_path = './data/descriptions_dataset.json'

if not os.path.exists(descriptions_dataset_path):
    raise FileNotFoundError(f"Не найден датасет : {descriptions_dataset_path}")

with open(descriptions_dataset_path, 'r', encoding='utf-8') as f:
    dataset = json.load(f)

corefPipeline = CorefPipeline(batch_size=4, udpipe_port=args.udpipe_port)

postpipeline_dataset = []
for book in dataset:
    sentences_count_on_prefix = []
    cnt = 0 
    merged_conllu = []
    merged_raw = []
    gold_descriptions = []
    for scene in book["Сцены"]:
        conllu = corefPipeline._process_text_local_udpipe(scene["Текст"])
        merged_conllu.append(conllu)
        merged_raw.append(scene["Текст"])
        cnt += conllu.count('sent_id')
        sentences_count_on_prefix.append(cnt)
        gold_descriptions.append(scene["ВизуальноеОписание"])
    merged_conllu = '\n'.join(merged_conllu)
    merged_raw = '\n'.join(merged_raw)
    # postpipeline_dataset.append((corefPipeline.get_clusters_info(text=merged_conllu, text_type='conllu'),
    #                              sentences_count_on_prefix,
    #                              gold_descriptions))
    postpipeline_dataset.append((corefPipeline.get_clusters_info(text=merged_raw, text_type='raw'),
                                 sentences_count_on_prefix,
                                 gold_descriptions))
corefPipeline.cleanup()

visExtractor = VisExtractor()

scores = []

for cluster_info, sentences_count_on_prefix, gold_descriptions in postpipeline_dataset:
    results = []
    for i in range(len(sentences_count_on_prefix)):
        # filtered_cluster_info = []
        prediction = {}
        for cluster_id in range(len(cluster_info)):
            filtered_sentences = []
            for sent_id, highlighted_sentence, sentence_mentions in cluster_info[cluster_id]["sentences"]:
                if sent_id < sentences_count_on_prefix[i]:
                    filtered_sentences.append((sent_id, highlighted_sentence, sentence_mentions))
            if not filtered_sentences:
                break
            filtered_cluster_info = {
                "mentions": cluster_info[cluster_id]["mentions"],
                "sentences": filtered_sentences
            }
            desc = visExtractor.generate_description(filtered_cluster_info, do_sample=False)

            desc_added = False
            for ment in cluster_info[cluster_id]["mentions"]:
                if ment not in prediction:
                    prediction[ment] = desc
                    desc_added = True
            if not desc_added:
                prediction[str(len(prediction))] = desc

        results.append(prediction)
    key_matching_scores, description_matching_scores = calculate_scores(gold_descriptions, results)
    scores.append((sentences_count_on_prefix, key_matching_scores, description_matching_scores))

print(scores)
with open("eval/pipeline_scores.json", "w") as file:
    json.dump(scores, file)

# results = []
# for i, text in enumerate(cumulative_texts):
#     prompt = text
#     clusters_info = corefPipeline.get_clusters_info(prompt)

#     prediction = {}
#     for cluster in clusters_info:
#         context = '. '.join(sentence for _, sentence, _ in cluster["sentences"])
#         mentions = ', '.join(cluster["mentions"])
#         user_prompt = f"Контекст: {context} | Сущность: {mentions} | Создай актуальное и точное визуальное описание."
#         messages = [
#             {"role": "system", "content": system_prompt},
#             {"role": "user", "content": user_prompt}
#         ]
#         text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
#         model_inputs = tokenizer([text], return_tensors="pt").to(inference_model.device)

#         generated_ids = inference_model.generate(
#             **model_inputs,
#             max_new_tokens=256,
#             temperature=0.1,
#         )
#         response = tokenizer.decode(generated_ids[0][len(model_inputs.input_ids[0]):], skip_special_tokens=True)
#         if cluster["mentions"]:
#             prediction[cluster["mentions"][0]] = response

#     results.append(prediction)

# key_matching_scores, description_matching_scores = calculate_scores(gold_descriptions, results)
# print()
# print("Semantic similarity score after key matching:", key_matching_scores)
# print("Average score after key matching:", sum(key_matching_scores) / len(key_matching_scores))
# print()
# print("Semantic similarity score after description matching:", description_matching_scores)
# print("Average score after description matching:", sum(description_matching_scores) / len(description_matching_scores))
