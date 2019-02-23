import sys
import math

TRAIN = './CSE447-HW1-Data/1b_benchmark.train.tokens'
DEV = './CSE447-HW1-Data/1b_benchmark.dev.tokens'
TEST = './CSE447-HW1-Data/1b_benchmark.test.tokens'
UNK = '@@UNK@@'
START = '@@START@@'
STOP = '@@STOP@@'


# a 2d list ans such that ans[] is sentences, ans[][] is tuple that represent groups
def grouptokens(tokens, vocab, groupsize):
    ans = []
    for line in tokens:
        line_group = []
        for i in range(len(line) - groupsize + 1):
            group = []
            for j in range(groupsize):
                group.append(line[i + j] if line[i + j] in vocab else UNK)
            line_group.append(tuple(group))
        ans.append(line_group)
    return ans


# given model and tokens and the correct format of unk, evaluate the perplexity
def eval_gram_perp(model, tokens, if_unk):
    log_prob = []
    for line in tokens:
        for word in line:
            if word not in model:
                if if_unk:
                    word = UNK
                else:
                    # if the combination is never seen before in training model, just return infinite
                    return 'infinite'
            log_prob.append((word, math.log2(model[word])))
    corpus_size = sum(map(len, tokens))
    average_likelihood = (1 / corpus_size) * sum(map(lambda f: f[1], log_prob))
    perplexity = 2 ** -average_likelihood
    return perplexity


# take a file path, tokenize the file and put it in 2d list
def process_tokens(file_path, padding):
    tokens = []
    with open(file_path, 'r') as f:
        for lines in f:
            tokenize_line = []
            for _ in range(padding):
                tokenize_line.append(START)
            for word in lines.split():
                tokenize_line.append(word)
            tokenize_line.append(STOP)
            tokens.append(tokenize_line)
    return tokens


# return a dictionary that contains the vocabulary of the corpus already Unkafied
# mapping to a count of such word appearing in the training data
def count_and_unkafy(tokens):
    rec = {}
    for lineToken in tokens:
        for word in lineToken:
            if word in rec:
                rec[word] = rec[word] + 1
            else:
                rec[word] = 1
    count = 0
    ans = {}
    for x in rec:
        if rec[x] < 3:
            count += rec[x]
        else:
            ans[x] = rec[x]
    ans[UNK] = count
    return ans


# return a model for unigram trained by the given tokens
def unigram(tokens):
    model = count_and_unkafy(tokens)
    total_count = sum(map(len, tokens))
    for word in model:
        model[word] = model[word] / total_count
    return model


# take a tuple, get rid of the last element and return the resulting tuple
def tuple_get_rid_last(group):
    element_list = []
    for i in range(len(group) - 1):
        element_list.append(group[i])
    return tuple(element_list)


# return a size-gram model trained on the given groups of tokens
def multigram(groups):
    model = {}
    history_count = {}
    for line in groups:
        for group in line:
            history = tuple_get_rid_last(group)
            model[group] = 1 if group not in model else model[group] + 1
            history_count[history] = 1 if history not in history_count else history_count[history] + 1
    for group in model:
        model[group] = model[group] / history_count[tuple_get_rid_last(group)]
    return model


# test the unigram on datasets by calculating their perplexity
def test_unigram():
    tokens = process_tokens(TRAIN, 0)
    model = unigram(tokens)
    print('unigram model perplexity for training set is: ' + str(eval_gram_perp(model, tokens, True)))
    print('unigram model perplexity for dev set is: ' + str(eval_gram_perp(model, process_tokens(DEV, 0), True)))
    print('unigram model perplexity for test set is: ' + str(eval_gram_perp(model, process_tokens(TEST, 0), True)))


# test the size-gram model by calculating their perplexity
def test_multigram(size):
    tokens = process_tokens(TRAIN, size - 1)
    vocab = count_and_unkafy(tokens)
    groups = grouptokens(tokens, vocab, size)
    model = multigram(groups)

    devtoken = process_tokens(DEV, size - 1)
    devgroup = grouptokens(devtoken, vocab, size)

    testtoken = process_tokens(TEST, size - 1)
    testgroup = grouptokens(testtoken, vocab, size)

    print(str(size) + 'gram model perplexity for training set is: ' + str(eval_gram_perp(model, groups, False)))
    print(str(size) + 'gram model perplexity for dev set is: ' + str(eval_gram_perp(model, devgroup, False)))
    print(str(size) + 'gram model perplexity for test set is: ' + str(eval_gram_perp(model, testgroup, False)))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        # print out the usage information
        print("Usage: python n-gram.py GRAM-SIZE(int 1 2 or 3)")
        sys.exit(0)
    gram_size = sys.argv[1]
    if gram_size == '1':
        test_unigram()
    elif gram_size == '2':
        test_multigram(2)
    elif gram_size == '3':
        test_multigram(3)
    elif gram_size == 'all':
        test_unigram()
        test_multigram(2)
        test_multigram(3)
