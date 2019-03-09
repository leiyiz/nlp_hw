// {
//     "dataset_reader": {
//         "type": "pos_tagger_read"
//     },
//     "train_data_path": "a4-data/a4-train.conllu",
//     "validation_data_path": "a4-data/a4-dev.conllu",
//     // "test_data_path": "data/en-ud-tweet-test.conllu",
//     "model": {
//         "type": "structured_perceptron_tagger",
//         "text_field_embedder": {
//             "type": "basic",
//             "token_embedders": {
//                 "tokens": {
//                     "type": "embedding",
//                     "embedding_dim": 300,
//                     "pretrained_file": "https://s3-us-west-2.amazonaws.com/allennlp/datasets/glove/glove.6B.300d.txt.gz",
//                     "trainable": true
//                 }
//             }
//         },
//         "encoder": {
//             "type": "lstm",
//             "bidirectional": true,
//             "input_size": 300,
//             "hidden_size": 64,
//             "num_layers": 3,
//             "dropout": 0.1
//         }
//     },
//     "iterator": {
//         "type": "basic",
//         "batch_size": 64
//     },
//     "trainer": {
//         "optimizer": {
//             "type": "adam",
//             "lr" : 0.001,
//             "weight_decay" : 0.001
//         },
//         "num_epochs": 20,
//         "cuda_device": 0,
//         "num_serialized_models_to_keep" : 0,
//         "validation_metric" : "+accuracy",
//         "learning_rate_scheduler" : {
//             "type" : "reduce_on_plateau",
//             "factor" : 0.3,
//             "patience" : 4
//         }
//     }
// }


{
    "dataset_reader": {
        "type": "pos_tagger_read",
        "token_indexers": {
            "tokens": {
                "type": "single_id",
                "lowercase_tokens": true
            },
            "elmo": {
                "type": "elmo_characters"
            }
        }
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
                    "embedding_dim": 300,
                    "pretrained_file": "https://s3-us-west-2.amazonaws.com/allennlp/datasets/glove/glove.6B.300d.txt.gz",
                    "trainable": true
                },
                "elmo": {
                    "type": "elmo_token_embedder",
                    "options_file": "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json",
                    "weight_file": "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5",
                    "do_layer_norm": false,
                    "dropout": 0.5
                }
            }
        },
        "encoder": {
            "type": "lstm",
            "bidirectional": true,
            "input_size": 1324,
            "hidden_size": 300,
            "num_layers": 4,
            "dropout": 0.1
        }
    },
    "iterator": {
        "type": "basic",
        "batch_size": 64
    },
    "trainer": {
        "optimizer": {
            "type": "adagrad",
            "lr" : 0.001,
            "weight_decay" : 0.001
            // "patience": 12
            // "momentum" : 0.1
        },
        "num_epochs": 20,
        "cuda_device": 0,
        "num_serialized_models_to_keep" : 0,
        "validation_metric" : "+accuracy",
        "learning_rate_scheduler" : {
            "type" : "reduce_on_plateau",
            "factor" : 0.3,
            "patience" : 4
        }
    }
}