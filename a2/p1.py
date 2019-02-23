from typing import Dict, Optional

import numpy
from overrides import overrides
import torch
import torch.nn.functional as F

from allennlp.common.checks import ConfigurationError
from allennlp.data import Vocabulary
from allennlp.modules import FeedForward, Seq2VecEncoder, TextFieldEmbedder
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn import util
from allennlp.training.metrics import CategoricalAccuracy


@Model.register("sentiment_classifier")
class SentimentClassifier(Model):
    """
        Parameters
        ----------
        vocab : ``Vocabulary``, required
            A Vocabulary, required in order to compute sizes for input/output projections.
        text_field_embedder : ``TextFieldEmbedder``, required
            Used to embed the ``tokens`` ``TextField`` we get as input to the model.
        token_encoder : ``Seq2VecEncoder``
            The encoder that we will use to convert the abstract to a vector.
        classifier_feedforward : ``FeedForward``
        initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
            Used to initialize the model parameters.
        regularizer : ``RegularizerApplicator``, optional (default=``None``)
            If provided, will be used to calculate the regularization penalty during training.
        """

    def __init__(self,
                 vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 token_encoder: Seq2VecEncoder,
                 classifier_feedforward: FeedForward,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(SentimentClassifier, self).__init__(vocab, regularizer)

        self.text_field_embedder = text_field_embedder
        self.num_classes = self.vocab.get_vocab_size("labels")
        self.token_encoder = token_encoder
        self.classifier_feedforward = classifier_feedforward
        self.metric = {
            "accuracy": CategoricalAccuracy(),
            "accuracy3": CategoricalAccuracy(top_k=3)
        }
        self.loss = torch.nn.CrossEntropyLoss()
        initializer(self)

    @overrides
    def forward(self,
                token: Dict[str, torch.Tensor],
                label: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        embedded_token = self.text_field_embedder(token)
        token_mask = util.get_text_field_mask(token)
        encoded_token = self.token_encoder(embedded_token, token_mask)

        logits = self.classifier_feedforward(encoded_token)
        probability = F.softmax(logits)

        output_dict = {"logits": logits, "probability": probability}

        if label is not None:
            loss = self.loss(logits, label.squeeze(-1))
            for metric in self.metric.values():
                metric(logits, label.squeeze(-1))
            output_dict["loss"] = loss

        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {metric_name: metric.get_metric(reset) for metric_name, metric in self.metric.items()}
