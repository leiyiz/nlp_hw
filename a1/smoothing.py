from n_gram import *


# build and return a interpolated model, here lambda is not yet applied to the model.
def interpolated_gram():
    unitokens = process_tokens(TRAIN, 0)
    unimodel = unigram(unitokens)

    bitokens = process_tokens(TRAIN, 1)
    vocab = count_and_unkafy(bitokens)
    bigroups = grouptokens(bitokens, vocab, 2)
    bimodel = multigram(bigroups)

    tritokens = process_tokens(TRAIN, 2)
    trigroups = grouptokens(tritokens, vocab, 3)
    trimodel = multigram(trigroups)

    return unimodel, bimodel, trimodel


# take a interpolate model, a model on the history of tri-gram and a pre-processed and grouped
# tokens array and the tuple of 3 lambda
def eval_interpolate_prep(model, first_two_model, tokens, lambs):
    log_prob = []
    lamb1_redis = lambs[0] + lambs[2] / 2
    lamb2_redis = lambs[1] + lambs[2] / 2
    for line in tokens:
        for word in line:
            prob = 0
            lamb1 = lambs[0]
            lamb2 = lambs[1]
            if word in model[2]:
                prob += lambs[2] * model[2][word]
                pass
            elif (word[0], word[1]) not in first_two_model:
                # re-distribute lambda for tri-gram to bi-gram and unigram if the tri word combination in test set is
                # never seen before
                lamb1 = lamb1_redis
                lamb2 = lamb2_redis
                pass
            if (word[1], word[2]) in model[1]:
                prob += lamb2 * model[1][(word[1], word[2])]
                pass
            if word[2] in model[0]:
                prob += lamb1 * model[0][word[2]]
            log_prob.append((word, math.log2(prob)))
    corpus_size = sum(map(len, tokens))
    average_likelihood = (1 / corpus_size) * sum(map(lambda f: f[1], log_prob))
    perplexity = 2 ** -average_likelihood
    return perplexity


# take a tuple of 3 lambdas, and (optionally) a boolean indicating if training set needs to be halved.
def test_interpogram(lambs, half=False):
    traintokens = process_tokens(TRAIN, 2)

    if half:
        traintokens = traintokens[:len(traintokens) // 2]
        pass

    vocab = count_and_unkafy(traintokens)
    traingroup = grouptokens(traintokens, vocab, 3)

    first_two_group = grouptokens(traintokens, vocab, 2)
    first_two_model = multigram(first_two_group)

    devtoken = process_tokens(DEV, 2)
    devgroup = grouptokens(devtoken, vocab, 3)

    test_token = process_tokens(TEST, 2)
    test_group = grouptokens(test_token, vocab, 3)

    interpo_model = interpolated_gram()

    print('interpolation gram model perplexity for training set is: ' + str(
        eval_interpolate_prep(interpo_model, first_two_model, traingroup, lambs)))
    print('interpolation gram model perplexity for dev set is: ' + str(
        eval_interpolate_prep(interpo_model, first_two_model, devgroup, lambs)))

    print('interpolation gram model perplexity for test set is: ' + str(
        eval_interpolate_prep(interpo_model, first_two_model, test_group, lambs)))


if __name__ == '__main__':
    if len(sys.argv) != 5:
        # print out the usage information.
        print('Usage: python smoothing.py Lambda1 lambda2 lambda3 half_the_training_set?(t/f)')
        sys.exit(0)
    lamb_nums = (float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]))
    if sys.argv[4] == 't':
        test_interpogram(lamb_nums, half=True)
    else:
        test_interpogram(lamb_nums)
