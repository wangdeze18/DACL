# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import logging
import sys, json
import numpy as np


def read_answers(filename):
    answers = {}
    with open(filename) as f:
        for line in f:
            line = line.strip()
            js = json.loads(line)
            answers[js['index']] = js['answers']
    return answers


def read_predictions(filename):
    predictions = {}
    with open(filename) as f:
        for line in f:
            line = line.strip()
            js = json.loads(line)
            predictions[js['index']] = js['answers']
    return predictions


def calculate_scores(answers, predictions, output_data_file):
    scores = {}
    for key in answers:
        #print(key)
        if key not in predictions:
            logging.error("Missing prediction for index {}.".format(key))
            sys.exit()

        ans_set = set(answers[key])
        pre_list = predictions[key]
        if len(ans_set) != len(pre_list):
            logging.error("Mismatch the number of answers for index {}.".format(key))
            sys.exit()

        score = 0
        have_find = 0
        #print(ans_set)
        for i in range(len(pre_list)):
            if pre_list[i] in ans_set:

                have_find += 1
                score += have_find / (i + 1)
        scores[int(key)] = score / len(pre_list)

    #scores = np.array(scores)
    #np.save(output_data_file, scores)
    with open(output_data_file,"w") as f:
        for i in sorted(scores):
            print(i)
            f.write(str(scores[i])+"\n")
    #result = {}
    #result['MAP'] = round(np.mean(scores), 4)
    return 0


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate leaderboard predictions for POJ-104 dataset.')
    parser.add_argument('--answers', '-a', help="filename of the labels, in txt format.")
    parser.add_argument('--predictions', '-p', help="filename of the leaderboard predictions, in txt format.")
    parser.add_argument("--output_data_file", '-o', type=str, required=True)

    args = parser.parse_args()
    answers = read_answers(args.answers)
    predictions = read_predictions(args.predictions)
    calculate_scores(answers, predictions, args.output_data_file)
    #print(scores)


if __name__ == '__main__':
    main()