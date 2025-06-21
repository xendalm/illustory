import requests
import tensorflow as tf
import os
from types import SimpleNamespace
import transformers
from LongCorpipe import corpipe24 as cor
from LongCorpipe.clusterer import merge_clusters
import json
from collections import defaultdict, Counter
import pymorphy2


class CorefPipeline:
    """
    A pipeline for performing coreference resolution on text using UDPipe and LongCorpipe.
    """

    # --- Class-level constants and initializations ---

    _BAD_TAGS = {'PRON', 'DET', 'PUNCT'}
    _morph = pymorphy2.MorphAnalyzer()

    def __init__(self, udpipe_port=8001, segment=2560, batch_size=8, threads=32):
        """
        Initializes the CorefPipeline, loading models and setting up configurations.
        """
        self._udpipe_port = udpipe_port
        corpipe_args = {
            "encoder": "google/mt5-large",
            "segment": segment,
            "right": 50,
            "zeros_per_parent": 2,
            "batch_size": batch_size,
            "threads": threads,
            "load": "LongCorpipe/corpipe24-corefud1.2-240906/",
            "depth": 5,
        }
        self._check_working_directory()

        self.corpipe_args = SimpleNamespace(**corpipe_args)
        self._configure_tensorflow()
        self.corpipe_tokenizer = self._load_tokenizer()
        self.corpipe_model = self._load_corpipe_model()

        self.corpipe_tags_map, self.corpipe_zdeprels_map = self._load_tag_and_deprel_maps()

    def _check_working_directory(self):
        """Ensures the script is run from the 'illustory' root directory."""
        current_dir = os.getcwd()
        if not current_dir.endswith('illustory'):
            raise RuntimeError(
                f"The script must be run from the 'illustory' root folder. Current folder: {current_dir}"
            )

    def _configure_tensorflow(self):
        """Configures TensorFlow to use GPU and sets inter-op parallelism threads."""
        try:
            gpu = tf.config.list_physical_devices('GPU')[0]
            tf.config.experimental.set_memory_growth(gpu, True)
            tf.config.threading.set_inter_op_parallelism_threads(self.corpipe_args.threads)
        except IndexError:
            print("No GPU found, TensorFlow will run on CPU.")
        except Exception as e:
            print(f"Error configuring TensorFlow: {e}")

    def _load_tokenizer(self):
        """Loads the Corpipe tokenizer and adds special tokens."""
        tokenizer = transformers.AutoTokenizer.from_pretrained(self.corpipe_args.encoder)
        tokenizer.add_special_tokens({
            "additional_special_tokens": [cor.Dataset.TOKEN_EMPTY] +
            [cor.Dataset.TOKEN_TREEBANK.format(i) for i in range(21)] +
            ([cor.Dataset.TOKEN_CLS] if tokenizer.cls_token_id is None else [])
        })
        return tokenizer

    def _load_corpipe_model(self):
        """Loads the Corpipe model."""
        with open(os.path.join(self.corpipe_args.load, "tags.txt"), mode="r") as tags_file:
            tags = [line.rstrip("\r\n") for line in tags_file]
        with open(os.path.join(self.corpipe_args.load, "zdeprels.txt"), mode="r") as zdeprels_file:
            zdeprels = [line.rstrip("\r\n") for line in zdeprels_file]

        return cor.Model(self.corpipe_tokenizer, tags, zdeprels, self.corpipe_args)

    def _load_tag_and_deprel_maps(self):
        """Loads tag and zdeprel maps for Corpipe."""
        with open(os.path.join(self.corpipe_args.load, "tags.txt"), mode="r") as tags_file:
            tags = [line.rstrip("\r\n") for line in tags_file]
        with open(os.path.join(self.corpipe_args.load, "zdeprels.txt"), mode="r") as zdeprels_file:
            zdeprels = [line.rstrip("\r\n") for line in zdeprels_file]

        return {tag: i for i, tag in enumerate(tags)}, {zdeprel: i for i, zdeprel in enumerate(zdeprels)}

    def _norm_mention(self, mention):
        """
        Normalizes a mention by converting its tokens to their normal form
        and filtering out bad tags.
        """
        tokens = [
            self._morph.parse(token.lower())[0].normal_form
            for token, _, pos in mention.info
            if pos not in self._BAD_TAGS
        ]
        tags = [pos for _, _, pos in mention.info if pos not in self._BAD_TAGS]

        if not any(tag in {"PROPN", "NOUN"} for tag in tags):
            return None

        return tuple(tokens)

    def _process_text_local_udpipe(self, text, port=None, output_format='conllu'):
        """
        Processes text using a local UDPipe server.
        """
        if not port:
            port = self._udpipe_port
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
                raise Exception(
                    f"Key 'result' not found in the server response: {result_json}"
                )
        except json.JSONDecodeError:
            raise Exception(f"Server returned invalid JSON. Response text: {response.text}")
        except requests.exceptions.RequestException as e:
            raise Exception(f"Error connecting to local UDPipe server: {e}")

    def get_clusters_info(self, text: str, text_type: str = 'raw'):
        """
        Performs coreference resolution on the given text and returns dict with clusters info.

        Args:
            text (str): The input text for coreference resolution.

        Returns:
            list: A list of dictionaries, where each dictionary represents a coreference cluster.
                  Each cluster contains 'mentions' (normalized forms of top 5 mentions) and
                  'sentences' (list of tuples: (sentence_id, highlighted_sentence, raw_mentions)).
            None: If an error occurs during the process.
        """
        temp_conllu_path = "temp_doc.conllu"
        try:
            if text_type == 'raw':
                print("Converting context to CoNLL-U...")
                conllu_text = self._process_text_local_udpipe(text)
            elif text_type == 'conllu':
                conllu_text = text
            else:
                raise ValueError(f"Unknown text type '{text_type}'")

            with open(temp_conllu_path, "w", encoding="utf-8") as f:
                f.write(conllu_text)
            print("CoNLL-U created successfully.")

            print("Running Corpipe for coreference resolution...")
            book = cor.Dataset(temp_conllu_path, self.corpipe_tokenizer, 0)
            generator = book.pipeline(
                self.corpipe_tags_map,
                self.corpipe_zdeprels_map,
                False,
                self.corpipe_args
            ).ragged_batch(self.corpipe_args.batch_size).prefetch(tf.data.AUTOTUNE)
            predicts = self.corpipe_model.predict(book, generator)

            if len(predicts) != 1:
                print(
                    f"WARN: There are {len(predicts)} documents in '{book._path}'! Should be 1"
                )

            predict = predicts[0]
            clusters = merge_clusters(predict)

            # Sort clusters by number of sentences they appear in, descending
            cluster_to_sentences = []
            for cluster in clusters:
                sentences_dict = defaultdict(list)
                for mention in cluster:
                    sent_id = mention.sent_id
                    sentences_dict[sent_id].append(mention)
                cluster_to_sentences.append(sentences_dict)
            cluster_to_sentences.sort(key=lambda x: -len(x))

            clusters_info = []
            total_sentences = sum([len(i) for i in cluster_to_sentences])
            threshold = total_sentences * 0.7
            current_sum = 0

            for sentences in cluster_to_sentences:
                if current_sum >= threshold:
                    break
                current_sum += len(sentences)
                cluster_candidates = Counter()
                cluster_sentences = []

                for sent_id, mentions in sentences.items():
                    # Get words for the current sentence
                    words = [flu[0] for flu in book.docs_flu[0][sent_id]]
                    
                    # Highlight mentions in the sentence
                    # It's important to make a copy if we're modifying it in place, 
                    # but here it's fine since 'words' is recreated per sentence.
                    highlighted_words = list(words) 
                    for mention in mentions:
                        start_id, end_id = mention.begin, mention.end
                        if 0 <= start_id < len(highlighted_words) and 0 <= end_id < len(highlighted_words):
                            highlighted_words[start_id] = "<" + highlighted_words[start_id]
                            highlighted_words[end_id] = highlighted_words[end_id] + ">"
                    
                    highlighted_sentence = " ".join(highlighted_words)

                    sentence_mentions = []
                    candidates_for_this_sentence = []
                    for mention in mentions:
                        mention_text = " ".join(word_info[0] for word_info in mention.info)
                        sentence_mentions.append(mention_text)
                        normalized_mention = self._norm_mention(mention)
                        if normalized_mention:
                            candidates_for_this_sentence.append(" ".join(normalized_mention))

                    cluster_sentences.append((sent_id, highlighted_sentence, sentence_mentions))
                    cluster_candidates.update(candidates_for_this_sentence)

                clusters_info.append({
                    "mentions": [item[0] for item in cluster_candidates.most_common(5)],
                    "sentences": cluster_sentences
                })
            print("Coreference resolution completed.")
            return clusters_info

        except Exception as e:
            print(f"Error during coreference resolution with Corpipe: {e}.")
            return None
        finally:
            if os.path.exists(temp_conllu_path):
                os.remove(temp_conllu_path)

    def cleanup(self):
        print("Cleaning up CorefPipeline resources...")
        
        del self.corpipe_model
        del self.corpipe_tokenizer
