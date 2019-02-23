# Usage:
## n_gram
- to run the code to evaluate n-gram model, pass the size of gram you want as an argument _(it has to be either **1, 2 or 3**)_. For example:
    ```bash
    python3 n_gram.py 1
    ```
    will evaluate the unigram model

## smoothing
- to tun the code to evaluate the interpolated model, pass three lambda of your choice in sequence of **lambda1 lambda2 lambda3** and a flag to tell the program if the training dataset needs to be halved _(**t** indicates that the training dataset needs to be halved and **everything else** means otherwise)_

    for example:
    ```bash
    python3 smoothing.py 0.1 0.3 0.6 f
    ```
    evaluate the interpolated model on the full training dataset with lambda values **lambda1 = 0.1**, **lambda2 = 0.3** and **lambda3 = 0.6**
- to change the threshold for UNKing, please navigate to the method **count_and_unkafy** and go to this line of code:
    ```python
    if rec[x] < 3:
    ```
    and change the 3 to whatever condition you want. More specifically, for unking words that appeared only once, change it to *2* and for unking words appeared less than 5 times, change the 3 to *5*