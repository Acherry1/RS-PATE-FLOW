import ast
from os import makedirs
import numpy as np
import pandas as pd
import torch
from loguru import logger
from data_now.experiment_factory import ExperimentFactory

from data_now.parameters import ExperimentParameters
from data_now.data_factory import DataFactory


def main(
        prms: ExperimentParameters,
        data_factory: DataFactory, data_name, image_size
):
    """
    This method trains a set of students as part of the PATE pipeline.

    For each previously executed voting, one student is trained on
    the public part of the dataset using the created labels.

    @param experiment_class: Class defining the training of one single student.
    @param prms: Parameters for the experiment,
        used for the training of all students.
    """

    train_student_fn = ExperimentFactory(prms.data.data_name).step_student

    combinations = [(voting_seed, aggregator)
                    for voting_seed in prms.pate.seeds2
                    for aggregator in prms.pate.aggregators]

    for i, (voting_seed, aggregator) in enumerate(combinations):
        student_dir = prms.student_dir(voting_seed=voting_seed,
                                       aggregator=aggregator)
        if (student_dir / 'model.h5').is_file() \
                and (student_dir / 'model.pickle').is_file():
            logger.info(
                f"Train student for voting_seed: {voting_seed}, aggregator: {aggregator}"
                f"has already been trained.")
            continue

        logger.info(
            f"Train student for voting_seed: {voting_seed}, aggregator: {aggregator}"
        )

        x_test, y_test = data_factory.data_test(seed=prms.pate.seed)
        x_valid, y_valid = data_factory.data_valid(seed=prms.pate.seed)
        voting_data = np.load(
            prms.voting_output_path(voting_seed=voting_seed,
                                    aggregator=aggregator))
        # unlabeled_data = np.load(
        #     prms.unlabeled_output_path(voting_seed=voting_seed,
        #                                aggregator=aggregator))
        features = voting_data['features']
        y_pred = voting_data['y_pred']
        y_true = voting_data['y_true']
        # unlabeled_features = unlabeled_data["unlabeled_features"]
        # unlabeled_targets = unlabeled_data["unlabeled_targets"]

        # load voting cost curve
        path_stats_voting = str(prms.resources.out_dir / 'stats_votings.csv')
        stats_voting = pd.read_csv(path_stats_voting, header=0)
        costs_curve = np.array(
            ast.literal_eval(stats_voting[
                                 (stats_voting['seed'] == prms.pate.seed)
                                 # & (stats_voting['seed2'] == voting_seed) &
                                 & (stats_voting['voting_seed'] == voting_seed) &
                                 (stats_voting['aggregator'] == aggregator) &
                                 (stats_voting['collector'] == prms.pate.collector) &
                                 (stats_voting['eps_short'] == str(prms.pate.eps_short)) &
                                 (stats_voting['distribution'] == str(prms.pate.distribution)) &
                                 (stats_voting['n_teachers'] == prms.pate.n_teachers) &
                                 (stats_voting['delta'] == prms.pate.delta) &
                                 (stats_voting['sigma'] == prms.pate.sigma) &
                                 (stats_voting['sigma1'] == prms.pate.sigma1) &
                                 (stats_voting['t'] == prms.pate.t)].iloc[0]['costs_curve']))

        if prms.pate.limits == ['budgets']:
            limits = [prms.pate.budgets]
        else:
            limits = prms.pate.limits
        for limit in limits:
            if isinstance(limit, int):
                n_limit = sum(np.cumsum(y_pred != -1) <= limit)
                n_labels = limit
            else:
                n_limit = sum(np.all(costs_curve <= np.array(limit), axis=1))
                n_labels = n_limit - sum(y_pred[:n_limit] == -1)
            costs = costs_curve[n_limit - 1]

            if n_labels < 2000:
                # select samples for which teachers responded
                response_filter = y_pred != -1
                # 取标签数据、无标签数据、验证数据
                x_train = features[response_filter]
                y_train = y_pred[response_filter]
                unlabeled_filter = y_pred == -1
                unlabeled_x_train = features[unlabeled_filter]
                unlabeled_y_train = y_pred[unlabeled_filter]

                # x_train_unlabeled = features[response_filter][n_labels:-n_labels]
                # y_train_unlabeled = y_pred[response_filter][n_labels:-n_labels]
                # x_valid = features[response_filter][-n_labels:-1]
                # y_valid = y_pred[response_filter][-n_labels:-1]
                # prms: ExperimentParameters,
                #         data_name,
                #         train_data: Tuple[np.array],
                #         test_data: Tuple[np.array],
                #         valid_data: Tuple[np.array],img_size,
                #         n_labels
                student, statistics = train_student_fn(
                    teacher_num=prms.pate.n_teachers,
                    prms=prms,
                    data_name=data_name,
                    train_data=(x_train, y_train),
                    unlabeled_train_data=(unlabeled_x_train, unlabeled_y_train),
                    test_data=(x_test, y_test),
                    valid_data=(x_valid, y_valid),
                    image_size=image_size,
                    n_labels=n_labels, seed=voting_seed
                )

                # save student and statistics together with parameters
                student_dir = prms.student_dir(voting_seed=voting_seed,
                                               aggregator=aggregator)
                makedirs(student_dir, exist_ok=True)
                # student.save(path=student_dir)
                torch.save(student.state_dict(), student_dir)
                stats_path = prms.resources.out_dir / 'stats_students.csv'
                statistics.update(prms.pate.__dict__)
                statistics.update({
                    'voting_seed': voting_seed,
                    'aggregator': aggregator,
                    'costs': costs,
                })
                pd.DataFrame(data=[statistics.values()],
                             columns=statistics.keys()).to_csv(
                    path_or_buf=stats_path,
                    mode='a',
                    header=not stats_path.is_file())
