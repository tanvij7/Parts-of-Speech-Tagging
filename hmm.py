import numpy as np
class POP(object):
    def __init__(self):
        self.vocab = None
        self.tags  = None

        self.A = None # transition matrix
        self.B = None # emission matrix

    def train(self, train_data):
        self.build_vocab(train_data)
        transition_count, emission_count, tag_count = self.get_counts(train_data)
        self.tags = sorted(list(tag_count.keys()))

        self.build_transition_martix(transition_count, tag_count, 1e-3)
        self.build_emission_matrix(emission_count, tag_count, 1e-3)

    def predict(self, test_ex):
        test_ex = test_ex.split()
        C, D = self.build_viterby_matrices(test_ex)
        seq = self.get_seq_of_pop(C, D)
        return seq

    #---------------------------------Training Helper Functions --------------------------------
    def build_vocab(self, train_data):
        word_freq = {}
        for i, line in enumerate(train_data):
            if line.strip():
                word, tag = line.split('\t')
                word_freq[word] = word_freq.get(word, 0) + 1

        # redusing scope of our vocabulary to simulate presence of unknown words
        # only the words which have aprpeaed in our corpus at least twice are kept
        vocab = [key for key, val in word_freq.items() if val >= 2]
        vocab.append('--unk--') # this tag will represent words which are not part of vocabulary
        vocab.sort()
        vocab = {word:idx for idx, word in enumerate(vocab)}
        self.vocab = vocab

    def read_line(self, line):
        vocab = self.vocab
        line  = line.strip()

        # empyt line indicates end of line in actual corpus
        # start of line token should be returned
        if not line:
            return '--n--', '--s--'
        else:
            word, tag = line.split('\t')

            # if word in our vocabulary return word with tag
            if word in vocab:
                return word, tag

            # word not in vocab => unkwon word
            else:
                return '--unk--', tag

    def get_counts(self, train_data):
        vocab = self.vocab

        # initialize dictionaries for different counts
        transition_count, emission_count, tag_count   = {}, {}, {}

        # initialize last_tag to start of sentense tag
        last_tag = '--s--'
        for line in train_data:
            word, tag = self.read_line(line)

            # Increment transition_count for transition from last tag to current tag
            transition_count[(last_tag, tag)] = transition_count.get((last_tag, tag), 0) + 1

            # Increment emission_count for emission from current tag to current word
            emission_count[(tag, word)] = emission_count.get((tag, word), 0) + 1

            # Increment count for current tag
            tag_count[tag] = tag_count.get(tag, 0) + 1

            # update last_tag
            last_tag = tag

        return transition_count, emission_count, tag_count

    def build_transition_martix(self, transition_count, tag_count, alpha):

        num_tags = len(tag_count)
        tag_list = sorted(list(tag_count.keys()))

        # Initialize transition matrix
        A = np.zeros((num_tags, num_tags))

        # row of transition matrix (from tag)
        for i in range(num_tags):
            # cilumn of transition matrix (to tag)
            for j in range(num_tags):

                # from => to, tag tag pair count
                from_tag, to_tag = tag_list[i], tag_list[j]
                count = transition_count.get((from_tag, to_tag), 0)

                # total number of times previous tag appeared
                # i.e. count of from tag
                count_of_from = tag_count.get(from_tag, 0)

                A[i,j] = (count + alpha )/ (count_of_from + alpha*num_tags)

        self.A = A

    def build_emission_matrix(self, emission_count, tag_count, alpha):
        vocab = self.vocab

        tag_list  = sorted(list(tag_count.keys()))
        word_list = sorted(list(vocab.keys()))
        num_tags  = len(tag_count)
        num_words = len(vocab)

        # Initialize emission matrix
        B = np.zeros((num_tags, num_words))

        # row of emission matrix (from tag)
        for i in range(num_tags):
            # column of emission matrix (to word)
            for j in range(num_words):

                # from => to, tag word pair count
                from_tag, to_word = tag_list[i], word_list[j]
                count = emission_count.get((from_tag, to_word), 0)

                # total number of times previous tag appeared
                # i.e. count of from tag
                count_of_from = tag_count.get(from_tag, 0)

                B[i,j] = (count + alpha )/ (count_of_from + alpha*num_words)

        self.B = B

    #---------------------------------Prediction Helper Functions --------------------------------

    def build_viterby_matrices(self, test):
        vocab    = self.vocab
        A, B     = self.A, self.B
        tag_list = self.tags

        word_list = sorted(list(vocab.keys()))
        num_tags  = len(tag_list)
        num_words = len(word_list)

        seq_len = len(test)

        # Initialize the matrices
        C, D = np.zeros((num_tags, seq_len)), np.zeros((num_tags, seq_len))

        # First column will be filled with transition from start tag
        start_tag_idx  = tag_list.index('--s--')
        first_word_idx = vocab.get(test[0], vocab['--unk--'])

        C[:,0] = (np.log(A[start_tag_idx,]) + np.log(B[:,first_word_idx]))
        D[:,0] = 0

        # loop for each word in test i.e column
        for j in range(1,len(test)):

            # loop over each tag i.e. row
            for i in range(num_tags):

                max_prob = -np.inf
                bext_tag = None

                # loop over provious tag probabilities
                for k in range(num_tags):

                    word_idx = vocab.get(test[j], vocab['--unk--'])
                    prob = C[k, j-1] + np.log(A[k, i]) + np.log(B[i, word_idx])

                    # probability for current k is better than best yet
                    if prob > max_prob:
                        max_prob = prob
                        bext_tag = k

                C[i,j] = max_prob
                D[i,j] = bext_tag

        return C, D

    def get_seq_of_pop(self, C, D):
        lg = C.shape[1]
        ans = np.zeros(lg, dtype=np.int)

        for i in range(lg)[::-1]:

            if i == lg-1:
                ans[i] = np.argmax(C[:,i])

            else:
                next_tag = ans[i+1]
                tag = D[next_tag, i+1]
                ans[i] = tag
        return ans
