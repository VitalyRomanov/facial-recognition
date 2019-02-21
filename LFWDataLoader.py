import numpy as np
import cv2

from pprint import pprint
from random import choice, sample
import pickle

import os
from os import path

class LFWDataLoader:
    """
    Reader for modified LFW dataset (Original: http://vis-www.cs.umass.edu/lfw/)
    It is assumemed that all the faces are cropped and aligned and have a square shape.
    The class implements the following functionality:
    - loading images
    - upsampling to the size suitable for CNN
    - passing images through CNN and obtaining CNN embeddings
    - caching CNN embeddings to avoid multiple calls of CNN on the same image
    - saving cached result to disk into 'embedding_cache.pkl'

    The class exposes next_minibatch that returns batch of images preprocessed by CNN.
    Images are read in Numpy format using cv2 module, upsampled to 299x299 pixels,
    and passed through CNN.

    The class assumes the presence of test set triplets sampled from the same datasset.
    Images from the test set are excluded from all minibatches
    """

    def_size = 299

    def __init__(self, data_path, emb_terminals, test_data_path):
        """

        :param data_path: Location of preprocessed dataset
        :param emb_terminals: Dictionary that contains keys 'input', 'output' and 'session'.
        'input' is tensorflow tensor for the input placeholder, 'output' is the terminal which
        returns images embedded with CNN, 'session' is an opened tensorflow session
        :param test_data_path: The location of test CSV file.
        """
        self.path = data_path
        self.emb_terminals = emb_terminals
        self._sample_pool = None
        # Create indexes
        self.indexes = self._file_index(data_path, self._get_people(data_path))

        # Load cache if exists
        self._emb_cache_file = "embedding_cache.pkl"
        if path.isfile(self._emb_cache_file):
            self._embedding_cache = pickle.load(open(self._emb_cache_file, "rb"))
        else:
            self._embedding_cache = dict()

        self.load_test_set(test_data_path)
        
        
    def _get_people(self, data_path):
        """
        Create a list of all people in the current dataset folder
        :param data_path:
        :return:
        """
        return [person for person in os.listdir(data_path) if path.isdir(path.join(data_path, person))]

    def _file_index(self, data_path, people):
        """
        Iterate through the dataset, filter people with less than 2 images, create indexes:
        - file_to_id: associates image path with a unique image ID
        - file_to_id: get file location by ID
        - id_to_pers: associate ID with a person identifier
        - pers_to_ids: associate person with the list of files
        :param data_path: location of the dataset
        :param people: list of people in the dataset
        :return: dictionary with indexes
        """

        all_files = {}

        # iterate thorough folders and associate people with filenames
        for person in people:
            person_path = path.join(data_path, person)
            files = [(person, path.join(person_path, file_)) for file_ in os.listdir(person_path) if file_[0] != "."]

            for file_ in os.listdir(person_path):
                if file_[0] != ".":
                    if person in all_files:
                        all_files[person].append(path.join(person_path, file_))
                    else:
                        all_files[person] = [path.join(person_path, file_)]

        # filter people with less than 2 photos
        for_pop = []
        for person in all_files:
            if len(all_files[person]) < 2:
                for_pop.append(person)

        for person in for_pop:
            all_files.pop(person, None)

        # convert dict into a flat list of tuples
        people_file_tuples = []
        for person in all_files:
            for file_ in all_files[person]:
                people_file_tuples.append((person, file_))

        people, files  = zip(*people_file_tuples)
        id_keys = list(range(len(files)))

        # Create indexes
        file_index = dict(zip(files, id_keys))
        file_inverted_index = dict(zip(id_keys, files))

        file_person_index = dict(zip(id_keys, people))
        person_file_index = {}
        for person, file_id in zip(people, id_keys):
            if person in person_file_index:
                person_file_index[person].append(file_id)
            else:
                person_file_index[person] = [file_id]

        return {
            'ids' : id_keys,
            'file_list' : files,
            'people_list' : people,
            'file_to_id' : file_index,
            'id_to_file' : file_inverted_index,
            'id_to_pers' : file_person_index,
            'pers_to_ids' : person_file_index
        }

    def next_minibatch(self, minibatch_size=256):
        """
        Generate minibatch if triplets (anchor, positive, negative).
        Corresponding images are preprocessed with CNN.
        :param minibatch_size:
        :return: ((batch_of_anchors, batch_of_positive, batch_of_negative), epoch_finished).
        epoch_finished is binary indication that the current epoch has lapsed
        batch_of are Numpy ndarrays
        """

        # For the first batch, initialize the epoch
        if not self._sample_pool:
            self._sample_pool = set(self.indexes['ids']) - self.test_ids

        # Sample anchors randomly and exclude anchors from rest minibatches
        # in the current epoch
        mb_anchors = sample(self._sample_pool, min(minibatch_size, len(self._sample_pool)))
        self._sample_pool -= set(mb_anchors)

        epoch_finished = len(self._sample_pool) == 0

        positive = self._get_positive_naive(mb_anchors)
        negative = self._get_negative_naive(mb_anchors)

        return self._get_batch_as_numpy((mb_anchors, positive, negative)), epoch_finished

    def _get_positive_naive(self, ids):
        """
        Get positive samples for listed ids. Positive samples are sampled uniformly
        :param ids:
        :return: list of positive ids
        """
        id2p = self.indexes['id_to_pers']
        p2id = self.indexes['pers_to_ids']

        positives = []
        for id_ in ids:
            positive = sample(p2id[id2p[id_]], 1)[0]
            while positive == id_:
                positive = sample(p2id[id2p[id_]], 1)[0]
            positives.append(positive)

        return positives

    def _get_negative_naive(self, ids):
        """
        Get negative samples for listed ids. Positive samples are sampled uniformly
        :param ids:
        :return: list of negative ids
        """
        id_keys = self.indexes['ids']
        id2p = self.indexes['id_to_pers']

        negatives = []
        for id_ in ids:
            negative = sample(id_keys, 1)[0]
            while id2p[negative] == id2p[id_]:
                negative = sample(id_keys, 1)[0]
            negatives.append(negative)

        return negatives

    def _get_batch_as_numpy(self, batch_ids):
        """
        Convert ids to CNN embeddigns
        :param batch_ids: tuple (anchors, positive, negative). Each element of the tuple is a list of IDs
        :return: a tuple of ndarrays of shape [?, cnn_embedding_size], where ? determined by the current batch size
        """
        mb_anchors, positives, negatives = batch_ids
        all_ids = set(mb_anchors) | set(positives) | set(negatives)

        self._cache(all_ids)

        emb_cache = self._embedding_cache

        get_embedding = lambda id_: emb_cache[id_]

        anchors = np.stack([get_embedding(id_) for id_ in mb_anchors], axis=0)
        pos = np.stack([get_embedding(id_) for id_ in positives], axis=0)
        neg = np.stack([get_embedding(id_) for id_ in negatives], axis=0)

        return anchors, pos, neg

    def _cache(self, ids):
        """
        Provided the list of IDs, load images from the dist, pass through CNN and shore in cache
        :param ids:
        :return: None
        """
        emb_cache = self._embedding_cache
        id2file = self.indexes['id_to_file']

        # Get ids that are not in cache
        for_caching = [id_ for id_ in ids if id_ not in emb_cache]

        if for_caching:

            # load all images for caching
            def_size = self.def_size # hardcoded to 299
            load_image = lambda id_: cv2.resize(cv2.imread(id2file[id_]), (def_size, def_size))
            all_imgs = np.stack(list(map(load_image, for_caching)), axis=0)

            # get tensorflow terminals
            input_ = self.emb_terminals['input']
            out_ = self.emb_terminals['output']
            sess = self.emb_terminals['session']

            # to avoid memory issues, cache images in small batches
            caching_batch_size = 50
            position = 0
        
            print("Caching new embeddings...{}/{}\r".format(position, len(for_caching)) , end="")

            while position < len(for_caching):
                print("Caching new embeddings...{}/{}\r".format(position, len(for_caching)) , end="")
                batch = all_imgs[position: min(position + caching_batch_size, len(for_caching)), ...]
                caching_batch = for_caching[position: min(position + caching_batch_size, len(for_caching))]

                embedded = sess.run(out_, {input_: batch})

                for ind, id_ in enumerate(caching_batch):
                    emb_cache[id_] = embedded[ind, :]

                position += caching_batch_size

            print("Caching new embeddings...done {} cached            ".format(len(emb_cache)))

            pickle.dump(emb_cache, open(self._emb_cache_file, "wb"))

    def generate_test_set(self):
        """
        Save a subset of dataset as test set
        :return:
        """
        sample_pool = set(self.indexes['ids'])
        
        test_anchors = sample(sample_pool, 400)

        positive = self._get_positive_naive(test_anchors)
        negative = self._get_negative_naive(test_anchors)

        id2file = self.indexes['id_to_file']

        id2p = lambda id_: "/".join(id2file[id_].split("/")[1:])

        with(open("test_set.csv", "w")) as test_sink:
            test_sink.write("Anchor,Positive,Negative\n")
            for a, p, n in zip(test_anchors, positive, negative):
                test_sink.write("{},{},{}\n".format(id2p(a), id2p(p), id2p(n)))


    def load_test_set(self, test_set_path):
        """
        Load test set from CSV file and store the test batch in self.test_batch
        :param test_set_path: path to CSV
        :return: None
        """

        f2id = self.indexes['file_to_id']
        data_path = self.path

        # read csv
        lines = open(test_set_path, "r").read().strip().split()
        a_path, p_path, n_path = zip(*list(map(lambda ln: tuple(ln.strip().split(",")), lines[1:])))

        get_id = lambda path_part: f2id[path.join(data_path, path_part)]

        anc = list(map(get_id, a_path))
        pos = list(map(get_id, p_path))
        neg = list(map(get_id, n_path))

        self.test_ids = set(anc) | set(pos) | set(neg)

        self.test_batch = self._get_batch_as_numpy((anc, pos, neg))




