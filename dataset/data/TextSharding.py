# coding=utf-8
# Copyright 2021 Intel Corporation. All rights reserved.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import defaultdict
from itertools import islice

import numpy as np

import multiprocessing
import statistics
from tqdm import tqdm


class Sharding:
    def __init__(
        self, input_files, output_name_prefix, n_training_shards, n_test_shards, fraction_test_set
    ):
        assert len(input_files) > 0, "The input file list must contain at least one file."
        assert n_training_shards > 0, "There must be at least one output shard."
        assert n_test_shards > 0, "There must be at least one output shard."

        self.n_training_shards = n_training_shards
        self.n_test_shards = n_test_shards
        self.fraction_test_set = fraction_test_set

        self.input_files = input_files

        self.output_name_prefix = output_name_prefix
        self.output_training_identifier = "training"
        self.output_test_identifier = "test"
        self.output_file_extension = ".txt"

        self.articles = {}  # key: integer identifier, value: list of articles
        self.sentences = {}  # key: integer identifier, value: list of sentences
        self.output_training_files = {}  # key: filename, value: list of articles to go into file
        self.output_test_files = {}  # key: filename, value: list of articles to go into file

        self.init_output_files()

    # Remember, the input files contain one article per line (the whitespace check is to skip extraneous blank lines)
    def load_articles(self):
        print("Start: Loading Articles")

        global_article_count = 0
        for input_file in self.input_files:
            print("input file:", input_file)
            with open(input_file, mode="r", newline="\n") as f:
                for i, line in enumerate(f):
                    if line.strip():
                        self.articles[global_article_count] = line.rstrip()
                        global_article_count += 1

        print("End: Loading Articles: There are", len(self.articles), "articles.")

    def segment_articles_into_sentences(self, segmenter):
        print("Start: Sentence Segmentation")
        if len(self.articles) is 0:
            self.load_articles()

        assert (
            len(self.articles) is not 0
        ), "Please check that input files are present and contain data."

        # TODO: WIP: multiprocessing (create independent ranges and spawn processes)
        use_multiprocessing = "queue"

        def chunks(data, n_processes=7):
            size = len(data)
            chunk_sz = int(size//n_processes)
            if size % n_processes != 0:
                chunk_sz += 1

            start_index = 0
            all_indices = np.arange(len(data))
            np.random.shuffle(all_indices)
            for i in range(0, n_processes):
                end_index = min(size, start_index + chunk_sz)
                yield {k: data[k] for k in all_indices[start_index: end_index]}
                start_index = end_index

        # TODO: WIP: multiprocessing (use manager dict for multiprocessing)
        if use_multiprocessing == "manager":
            manager = multiprocessing.Manager()
            return_dict = manager.dict()
            jobs = []
            n_processes = 16

            def work(articles, return_dict):
                sentences = {}
                for i, article in enumerate(articles):
                    sentences[i] = segmenter.segment_string(articles[article])

                    if i % 5000 == 0:
                        print("Segmenting article", i)

                return_dict.update(sentences)

            for item in chunks(self.articles, len(self.articles)):
                p = multiprocessing.Process(target=work, args=(item, return_dict))

                # Busy wait
                while len(jobs) >= n_processes:
                    pass

                jobs.append(p)
                p.start()

            for proc in jobs:
                proc.join()

        elif use_multiprocessing == "queue":
            n_processes = 16
            work_queue = multiprocessing.Queue()
            jobs = []

            def child_work(articles, process_index, work_queue):
                for i, article in enumerate(articles):
                    sentence = segmenter.segment_string(articles[article])
                    work_queue.put((process_index, article, sentence))
                work_queue.put((process_index, "Done"))

            for item in chunks(self.articles, n_processes):
                print("chunk size = ", len(item))
                p = multiprocessing.Process(target=child_work, args=(item, len(jobs), work_queue))

                jobs.append(p)
                print("Number of Jobs = ", len(jobs))
                p.start()
                print(f"Process {len(jobs)-1} Started")

            done_tasks = 0
            n_sentences = 0
            while True:
                msg = work_queue.get()
                if msg[1] == "Done":
                    done_tasks += 1
                    print(f"Completed Process: {msg[0]}")
                    if done_tasks == n_processes:
                        break
                else:
                    self.sentences[msg[1]] = msg[2]
                    if n_sentences % 5000 == 0:
                        print("Segmenting article", n_sentences)
                    n_sentences += 1

            print("Joining Jobs")
            for proc in jobs:
                proc.join()

        else:  # serial option
            for i, article in enumerate(self.articles):
                self.sentences[i] = segmenter.segment_string(self.articles[article])

                if i % 5000 == 0:
                    print("Segmenting article", i)

        print("End: Sentence Segmentation")

    def init_output_files(self):
        print("Start: Init Output Files")
        assert (
            len(self.output_training_files) is 0
        ), "Internal storage self.output_files already contains data. This function is intended to be used by the constructor only."
        assert (
            len(self.output_test_files) is 0
        ), "Internal storage self.output_files already contains data. This function is intended to be used by the constructor only."

        for i in range(self.n_training_shards):
            name = (
                self.output_name_prefix
                + self.output_training_identifier
                + str(i)
                + self.output_file_extension
            )
            self.output_training_files[name] = []

        for i in range(self.n_test_shards):
            name = (
                self.output_name_prefix
                + self.output_test_identifier
                + str(i)
                + self.output_file_extension
            )
            self.output_test_files[name] = []

        print("End: Init Output Files")

    def get_sentences_per_shard(self, shard):
        result = 0
        for article_id in shard:
            result += len(self.sentences[article_id])

        return result

    def distribute_articles_over_shards(self):
        print("Start: Distribute Articles Over Shards")
        assert (
            len(self.articles) >= self.n_training_shards + self.n_test_shards
        ), "There are fewer articles than shards. Please add more data or reduce the number of shards requested."

        articles = np.fromiter(self.articles.keys(), dtype=np.int32, count=len(self.articles))
        # shuffle articles to distribute them evenly over the shards
        np.random.shuffle(articles)

        articles_per_test_shard = int(self.fraction_test_set * len(articles)) // self.n_test_shards
        total_test_articles = articles_per_test_shard * self.n_test_shards
        articles_per_training_shard = (len(articles) - total_test_articles) // self.n_training_shards

        i = 0
        for article_id in tqdm(articles):
            if i < total_test_articles:
                shard_id = i // articles_per_test_shard
                if shard_id >= self.n_test_shards:
                    continue
                key = (
                    self.output_name_prefix
                    + self.output_test_identifier
                    + str(shard_id)
                    + self.output_file_extension
                )
                self.output_test_files[key].append(article_id)
            else:
                shard_id = (i - total_test_articles) // articles_per_training_shard
                if shard_id >= self.n_training_shards:
                    continue
                key = (
                    self.output_name_prefix
                    + self.output_training_identifier
                    + str(shard_id)
                    + self.output_file_extension
                )
                self.output_training_files[key].append(article_id)
            i += 1

        for shard in self.output_training_files:
            print(
                "Training shard:", self.get_sentences_per_shard(self.output_training_files[shard])
            )

        for shard in self.output_test_files:
            print("Test shard:", self.get_sentences_per_shard(self.output_test_files[shard]))

        print("End: Distribute Articles Over Shards")

    def write_shards_to_disk(self):
        print("Start: Write Shards to Disk")
        for shard in self.output_training_files:
            self.write_single_shard(shard, self.output_training_files[shard])

        for shard in self.output_test_files:
            self.write_single_shard(shard, self.output_test_files[shard])

        print("End: Write Shards to Disk")

    def write_single_shard(self, shard_name, shard):
        with open(shard_name, mode="w", newline="\n") as f:
            for article_id in shard:
                for line in self.sentences[article_id]:
                    f.write(line + "\n")

                f.write("\n")  # Line break between articles


try:
    import nltk

    nltk.download("punkt")
except ModuleNotFoundError or ImportError as e:
    print("nltk is required for sharding. please install before running.")


class NLTKSegmenter:
    def segment_string(self, article):
        return nltk.tokenize.sent_tokenize(article)
