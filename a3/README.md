##A3 instructions
allennlp predict simple_result/model.tar.gz data/en-ud-tweet-test.conllu --output-file test.tsv --use-dataset-reader --predictor twitter_tagger_pred --include-package twitter_tagger --cuda-device 0
