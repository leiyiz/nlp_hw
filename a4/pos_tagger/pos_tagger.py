from typing import Dict, Optional, List, Any

from overrides import overrides
import torch
from torch.nn.modules.linear import Linear

from allennlp.common.checks import check_dimensions_match, ConfigurationError
from allennlp.data import Vocabulary
from allennlp.modules import Seq2SeqEncoder, TimeDistributed, TextFieldEmbedder
from allennlp.modules import FeedForward
from allennlp.modules.conditional_random_field import allowed_transitions
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
import allennlp.nn.util as util
from allennlp.training.metrics import CategoricalAccuracy, SpanBasedF1Measure

from pos_tagger.structured_perceptron import StructuredPerceptron


@Model.register("structured_perceptron_tagger")
class POSTagger(Model):
    """
    Parameters
    ----------
    vocab : ``Vocabulary``, required
        A Vocabulary, required in order to compute sizes for input/output
        projections.
    text_field_embedder : ``TextFieldEmbedder``, required
        Used to embed the tokens ``TextField`` we get as input to the model.
    """

    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder,
                 feedforward: Optional[FeedForward] = None) -> None:
        super().__init__(vocab)

        self.text_field_embedder = text_field_embedder
        self.label_namespace = 'labels'
        self.num_tags = self.vocab.get_vocab_size(self.label_namespace)
        self.binary_potentials = torch.nn.Parameter(torch.zeros(self.num_tags, self.num_tags),requires_grad=True)

        # POSTagger uses the StructuredPerceptron class.
        self.structured_perceptron = StructuredPerceptron()
        self._feedforward = feedforward
        self._device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.encoder = encoder
        if feedforward is not None:
            output_dim = feedforward.get_output_dim()
            self.tag_proj_layer = TimeDistributed(Linear(output_dim, self.num_tags))
        else:
            output_dim = self.encoder.get_output_dim()
            self.tag_proj_layer = TimeDistributed(Linear(output_dim, self.num_tags))
        self.metrics = {
            "accuracy": CategoricalAccuracy(),
        }

    @overrides
    def forward(self,  # type: ignore
                tokens: Dict[str, torch.LongTensor],
                tags: torch.LongTensor = None,
                metadata: List[Dict[str, Any]] = None,
                # pylint: disable=unused-argument
                **kwargs) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        tokens : ``Dict[str, torch.LongTensor]``, required
            The output of ``TextField.as_array()``, which should typically be
            passed directly to a ``TextFieldEmbedder``. This output is a
            dictionary mapping keys to ``TokenIndexer`` tensors.  At its most
            basic, using a ``SingleIdTokenIndexer`` this is: ``{"tokens":
            Tensor(batch_size, num_tokens)}``. This dictionary will have the
            same keys as were used for the ``TokenIndexers`` when you created
            the ``TextField`` representing your sequence.  The dictionary is
            designed to be passed directly to a ``TextFieldEmbedder``, which
            knows how to combine different word representations into a single
            vector per token in your input.
        tags : ``torch.LongTensor``, optional (default = ``None``)
            A torch tensor representing the sequence of integer gold class
            labels of shape ``(batch_size, num_tokens)``.
        metadata : ``List[Dict[str, Any]]``, optional, (default = None)
            metadata containg the original words in the sentence to be tagged
            under a 'words' key.

        Returns
        -------
        An output dictionary consisting of:

        unary_potentials : ``torch.FloatTensor``
            The unary potentials.
        binary_potentials : ``torch.FloatTensor``
            The binary potentials.
        mask : ``torch.LongTensor``
            The text field mask for the input tokens
        tags : ``List[List[int]]``
            The predicted tags.
        loss : ``torch.FloatTensor``, optional
            A scalar loss to be optimised. Only computed if gold label ``tags``
            are provided.
        """
        embedded_text_input = self.text_field_embedder(tokens)
        batch_size, sequence_length, _ = embedded_text_input.size()
        mask = util.get_text_field_mask(tokens)
        encoded_text = self.encoder(embedded_text_input, mask)

        # batch_size = tags.shape[0]
        # sequence_length = mask.shape[1]

        if self._feedforward is not None:
            encoded_text = self._feedforward(encoded_text)

        unary_potentials = self.tag_proj_layer(encoded_text)
        best_paths = self.structured_perceptron.get_tags(unary_potentials, self.binary_potentials, mask)
        pred_tags = [x for x, y in best_paths]

        output_dict = {"unary_potentials": unary_potentials,
                       "binary_potentials": self.binary_potentials,
                       "mask": mask,
                       "tags": pred_tags}

        if tags is not None:
            loss = self.structured_perceptron.forward(unary_potentials, self.binary_potentials, tags, pred_tags, mask)
            loss = loss.to(self._device)
            output_dict["loss"] = loss

            class_probabilities = torch.zeros((batch_size, sequence_length, self.num_tags)).to(self._device)

            for i, instance_tags in enumerate(pred_tags):
                for j, tag_id in enumerate(instance_tags):
                    class_probabilities[i, j, tag_id] = 1

            for metric in self.metrics.values():
                metric(class_probabilities, tags, mask.float())

        if metadata is not None:
            output_dict["words"] = [x["words"] for x in metadata]

        return output_dict

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Converts the tag ids to the actual tags.
        ``output_dict["tags"]`` is a list of lists of tag_ids, so we use an
        ugly nested list comprehension.
        """
        output_dict["tags"] = [
            [self.vocab.get_token_from_index(tag, namespace=self.label_namespace)
             for tag in instance_tags]
            for instance_tags in output_dict["tags"]
        ]

        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics_to_return = {metric_name: metric.get_metric(reset) for
                             metric_name, metric in self.metrics.items()}

        return metrics_to_return
