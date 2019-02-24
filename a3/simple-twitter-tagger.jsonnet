{
    "dataset_reader": {
        "type": "twitter_tagger_read"
    },
    "train_data_path": "data/en-ud-tweet-train.conllu",
    "validation_data_path": "data/en-ud-tweet-dev.conllu",
    "test_data_path": "data/en-ud-tweet-test.conllu",
    "model": {
        "type": "simple_tagger",
        "text_field_embedder": {
            "type": "basic",
            "token_embedders": {
                "tokens": {
                    "type": "embedding",
                    "embedding_dim": 64
                }
            }
        },
        "encoder": {
            "type": "lstm",
            "input_size": 64,
            "hidden_size": 64
        }
    },
    "iterator": {
        "type": "basic"
    },
    "trainer": {
        "optimizer": {
            "type": "adam"
        },
        "num_epochs": 50,
        "cuda_device": 0
    }
}