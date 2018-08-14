import numpy as np
from pathlib import Path
import pickle
import fire

from lm.preprocessing import WikiTextTokenizer


def tokenize_train(train_path, min_frequency, vocab_size):
    """
    Load, preprocess and tokenize the wikitext 103 training dataset.
    In addition, a word2int dictonary is generated.
    """
    print('Tokenizing train dataset')
    wikitext = WikiTextTokenizer(filepaths=[train_path])

    # tokenize. This takes a while
    tokenized_wikitext = wikitext.tokenize()

    # turn the result into an array
    tw_ar = np.asarray(tokenized_wikitext)
    # we can begin by saving the entire dataset
    np.save('wikitext_train_str_tokens.npy', tw_ar)

    # now, to find the frequency of words
    unique_tokens, count = np.unique(tw_ar, return_counts=True)

    # get the ordered indices
    sort_indices = count.argsort()[::-1]
    sorted_tokens = unique_tokens[sort_indices]
    sorted_counts = count[sort_indices]

    # make sure all my selected vocab has at least min_frequency words
    assert sorted_counts[vocab_size] >= min_frequency

    # we will now add the unknown and padding tokens
    sorted_tokens = np.insert(sorted_tokens, 0, '_unk_')
    sorted_tokens = np.insert(sorted_tokens, 0, '_pad_')

    # word2int
    word2int = {token: idx for idx, token in enumerate(sorted_tokens[:(vocab_size + 2)])}

    unknown_int = word2int['_unk_']

    # now, we can turn tw_ar into an array of ints
    tw_ints = [word2int.get(tok, unknown_int) for tok in tw_ar]

    np.save('wikitext_train_int_tokens.npy', np.asarray(tw_ints))

    # finally, save the dictionary
    dict_path = Path('word2int.pickle')
    with dict_path.open(mode='wb') as file:
        pickle.dump(word2int, file, protocol=pickle.HIGHEST_PROTOCOL)

    return word2int


def tokenize_valtest(val_path, test_path, word2int):

    print('Tokenizing val/test dataset')
    wikitext = WikiTextTokenizer(filepaths=[val_path, test_path])

    # tokenize. This takes a while
    tokenized_wikitext = wikitext.tokenize()

    # turn the result into an array
    tw_ar = np.asarray(tokenized_wikitext)
    # we can begin by saving the entire dataset
    np.save('wikitext_valtest_str_tokens.npy', tw_ar)

    # now, we can turn tw_ar into an array of ints
    unknown_int = word2int['_unk_']
    tw_ints = [word2int.get(tok, unknown_int) for tok in tw_ar]

    np.save('wikitext_valtest_int_tokens.npy', np.asarray(tw_ints))


def tokenize_wikitext(train_path=Path('wikitext-103-raw/wiki.train.raw'),
                      val_path=Path('wikitext-103-raw/wiki.valid.raw'),
                      test_path=Path('wikitext-103-raw/wiki.test.raw'),
                      min_frequency=2, vocab_size=60000):

    word2int = tokenize_train(train_path, min_frequency, vocab_size)
    tokenize_valtest(val_path, test_path, word2int)


if __name__ == '__main__':
    fire.Fire(tokenize_wikitext)
