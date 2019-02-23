// This incomplete configuration contains settings necessary (but not sufficient)
// for training a neural language model that is comparable to the ngram LM in
// the first two portions of CSE 447 / 547M Winter 2019 A1.
// Note that the given configuration for the "vocabulary" and "dataset_reader" may not be complete.
// Part of this config is taken from https://github.com/allenai/allennlp/blob/088f0bb/training_config/bidirectional_language_model.jsonnet

{
    "vocabulary": {
        "tokens_to_add": {
            "tokens": ["<S>", "</S>"]
        },
        "min_count": {
            "tokens": 3
        }
    },
    "dataset_reader": {
        "type": "simple_language_modeling",
        "start_tokens": ["<S>"],
        "end_tokens": ["</S>"]
    }
}