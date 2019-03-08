{
    "dataset_reader": {
        "type": "pos_tagger_read"
    },
    "train_data_path": "a4-data/a4-train.conllu",
    "validation_data_path": "a4-data/a4-dev.conllu",
    // "test_data_path": "data/en-ud-tweet-test.conllu",
    "model": {
        "type": "structured_perceptron_tagger",
        "text_field_embedder": {
            "type": "basic",
            "token_embedders": {
                "tokens": {
                    "type": "embedding",
                    "embedding_dim": 512
                }
            }
        },
        "encoder": {
            "type": "lstm",
            "bidirectional": true,
            "input_size": 512,
            "hidden_size": 64,
            "num_layers": 3,
            "dropout": 0.1
        }
    },
    "iterator": {
        "type": "basic",
        "batch_size": 32
    },
    "trainer": {
        "optimizer": {
            "type": "adam",
            "lr" : 0.01,
            "weight_decay" : 0.0001
            // "momentum" : 0.1
        },
        "num_epochs": 20,
        "cuda_device": 0
    }
}