from typing import Dict
from overrides import overrides

from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import TextField, SequenceLabelField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.token_indexers.token_indexer import TokenIndexer
from allennlp.data.tokenizers import WordTokenizer, Token
from allennlp.data.tokenizers.tokenizer import Tokenizer


@DatasetReader.register('twitter_tagger_pred')
class SentimentPlaintextReader(DatasetReader):
    def __init__(self,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 use_subtrees: bool = False,
                 granularity: str = '2-class') -> None:
        super().__init__()
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = (token_indexers or
                                {'tokens': SingleIdTokenIndexer()})

    @overrides
    def _read(self, path: str):
        with open(path, 'r') as f:
            lines = f.readlines()

        pair = [[], []]
        for line in lines:
            words = line.strip().split()
            if len(words) == 0:
                instance = self.text_to_instance()
                yield instance
                pair = [[], []]
                pass
            else:
                if words[0].isdigit():
                    pair[0].append(words[1])
                    pair[1].append(words[3])
                    pass
                pass

    @overrides
    def text_to_instance(self, words, tags=None) -> Instance:
        tokens = [Token(word) for word in words]
        token_field = TextField(tokens, self._token_indexers)
        res = {'tokens': token_field}
        if tags is not None:
            tags_field = SequenceLabelField(labels=tags, sequence_field=token_field)
            res['tags'] = tags_field
            pass
        return Instance(res)
