{
    "dataset_reader": {
        "type": "twitter_tagger_read"
    },
    "train_data_path": "data/en-ud-tweet-train.conllu",
    "validation_data_path": "data/en-ud-tweet-dev.conllu",
    "test_data_path": "data/en-ud-tweet-test.conllu",
    "model": {
        "type": "crf_tagger",
        "text_field_embedder": {
            "type": "basic",
            "token_embedders": {
                "tokens": {
                    "type": "embedding",
                    "embedding_dim": 256
                }
            }
        },
        "encoder": {
            "type": "lstm",
            "bidirectional": true,
            "input_size": 256,
            "hidden_size": 256,
            "num_layers": 1,
            "dropout": 0.2
        }
    },
    "iterator": {
        "type": "basic",
        "batch_size": 16
    },
    "trainer": {
        "optimizer": {
            "type": "adam"
        },
        "num_epochs": 25,
        "cuda_device": 0
    }
}