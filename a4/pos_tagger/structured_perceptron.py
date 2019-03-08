"""
Structured perceptron implementation.
"""
from typing import List, Tuple, Dict, Optional

import torch

from allennlp.common.checks import ConfigurationError


class StructuredPerceptron(torch.nn.Module):
    def __init__(self) -> None:
        self._device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        super().__init__()

    def _score(self,
               unary_potentials: torch.Tensor,
               binary_potentials: torch.Tensor,
               tags: torch.Tensor,
               mask: torch.LongTensor) -> torch.Tensor:
        """
        Computes the score. Returns a Tensor of size [batch_size] where each
        element is the score for the sequence at that index.
        """
        batch_size, max_seq_length, num_tags = unary_potentials.size()
        score = torch.zeros([batch_size])
        for sentence_num in range(batch_size):
            curr_unary_potentials = unary_potentials[sentence_num]
            curr_tags = tags[sentence_num]
            curr_mask = mask[sentence_num]

            prev_tag = curr_tags[0]
            curr_score = curr_unary_potentials[0][prev_tag]
            for i in range(1, max_seq_length):
                if curr_mask[i] is 0:
                    break
                temp_tag = curr_tags[i]
                curr_score += curr_unary_potentials[i][temp_tag] + binary_potentials[prev_tag][temp_tag]
                prev_tag = temp_tag
            score[sentence_num] = curr_score

        return score.to(self._device)

    def forward(self,
                unary_potentials: torch.Tensor,
                binary_potentials: torch.Tensor,
                tags: torch.Tensor,
                predicted_tags: List,
                mask: torch.ByteTensor) -> torch.Tensor:
        # pylint: disable=arguments-differ
        predicted_tags = torch.nn.utils.rnn.pad_sequence(
            list(torch.tensor(tags) for tags in predicted_tags),
            batch_first=True).to(self._device)
        gold_scores = self._score(unary_potentials, binary_potentials, tags, mask)
        pred_scores = self._score(unary_potentials, binary_potentials, predicted_tags, mask)

        # Hinge loss.
        losses = pred_scores - gold_scores
        losses = torch.max(losses, torch.FloatTensor([0]).expand_as(losses).to(self._device))
        loss = torch.sum(losses)

        return loss

    def get_tags(self,
                 unary_potentials: torch.Tensor,
                 binary_potentials: torch.Tensor,
                 mask: torch.Tensor) -> List[Tuple[List[int], float]]:
        """
        Get the most likely tags for the given inputs.
        """
        _, max_seq_length, num_tags = unary_potentials.size()

        # Get the tensors out of the variables
        unary_potentials, mask = unary_potentials.data, mask.data

        start_tag = num_tags
        end_tag = num_tags + 1

        aug_binary_potentials = torch.Tensor(num_tags + 2, num_tags + 2).fill_(-10000.)

        # set the current transition matrix to the learned transitions
        aug_binary_potentials[:num_tags, :num_tags] = binary_potentials

        # Since we're not learning start and end transitions, we will set
        # them all to the same low potential.
        aug_binary_potentials[start_tag, :num_tags] = -10000.0
        aug_binary_potentials[:num_tags, end_tag] = -10000.0

        best_paths = []
        # Pad the max sequence length by 2 to account for start_tag + end_tag.
        tag_sequence = torch.Tensor(max_seq_length + 2, num_tags + 2)

        for prediction, prediction_mask in zip(unary_potentials, mask):
            sequence_length = torch.sum(prediction_mask)

            # Start with everything totally unlikely
            tag_sequence.fill_(-10000.)
            # At timestep 0 we must have the START_TAG
            tag_sequence[0, start_tag] = 0.
            # At steps 1, ..., sequence_length we just use the incoming
            # prediction
            tag_sequence[1:(sequence_length + 1), :num_tags] = \
                prediction[:sequence_length]
            # And at the last timestep we must have the END_TAG
            tag_sequence[sequence_length + 1, end_tag] = 0.

            # We pass the tags and the transitions to ``decode``.
            path, score = \
                self.decode(tag_sequence[:(sequence_length + 2)],
                            aug_binary_potentials)
            # Get rid of START and END sentinels and append.
            path = path[1:-1]
            best_paths.append((path, score.item()))

        return best_paths

    def decode(self,
               unary_potentials: torch.Tensor,
               binary_potentials: torch.Tensor):
        """
        Perform decoding in log space over a sequence given a matrix specifying
        unary potentials for possible tags per timestep and a transition matrix
        specifying pairwise (transition) potentials between tags.

        This is where you should implement the decoding algorithm you derived
        in A4 section 1.

        Parameters
        ----------
        unary_potentials : torch.Tensor, required.
            A tensor of shape (sequence_length, num_tags) representing unary
            potentials for a set of tags over a given sequence.
        binary_potentials : torch.Tensor, required.
            A tensor of shape (num_tags, num_tags) representing the binary
            potentials for transitioning between a given pair of tags.

        Returns
        -------
        path : List[int]
            The tag indices of the maximum likelihood tag sequence.
        score : torch.Tensor
            The score of the path.
        """
        sequence_len, _ = list(unary_potentials.size())

        path_scores = []
        back_pointer = []

        path_scores.append(unary_potentials[0, :])

        for step in range(1, sequence_len):
            sum_potential_heart = path_scores[step - 1].unsqueeze(-1) + binary_potentials \
                                  + unary_potentials[step, :]
            scores, paths = torch.max(sum_potential_heart, 0)
            # path_scores.append(unary_potentials[step, :] + scores.squeeze())
            # TODO: try remove the squeeze() here
            path_scores.append(scores.squeeze())
            back_pointer.append(paths.squeeze())

        score, best_path = torch.max(path_scores[-1], 0)
        path = [int(best_path.numpy())]
        for best_step in reversed(back_pointer):
            path.append(int(best_step[path[-1]]))
        path.reverse()
        return path, score
