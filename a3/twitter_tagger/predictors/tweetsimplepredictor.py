from overrides import overrides

from allennlp.common.util import JsonDict
from allennlp.data import DatasetReader, Instance
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter
from allennlp.models import Model
from allennlp.predictors.predictor import Predictor


@Predictor.register('twitter_tagger_pred')
class TweetSimplePredictor(Predictor):

    @overrides
    def dump_line(self, outputs: JsonDict):
        words = outputs['words']
        tags = outputs['tags']
        sentence = ' '.join(words)
        tag_line = ' '.join(tags)
        return sentence + '\t' + tag_line + '\n'
