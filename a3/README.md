##A3 instructions
allennlp predict simple.tar.gz data/en-ud-tweet-test.conllu --output-file simple-tagger-output.tsv --use-dataset-reader --predictor twitter_tagger_pred --include-package twitter_tagger --cuda-device 0


