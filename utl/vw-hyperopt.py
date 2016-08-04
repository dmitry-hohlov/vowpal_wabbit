#!/usr/bin/env python
# coding: utf-8

"""
Github version of hyperparameter optimization for Vowpal Wabbit via hyperopt
"""

__author__ = 'kurtosis'

from hyperopt import hp, fmin, tpe, rand, Trials, STATUS_OK
from sklearn.metrics import roc_curve, auc, log_loss, precision_recall_curve
import numpy as np
from datetime import datetime as dt
import subprocess, shlex, os
from math import exp, log
import argparse
import re
import logging
import json
import matplotlib
from matplotlib import pyplot as plt
try:
    import seaborn as sns
except ImportError:
    print ("Warning: seaborn is not installed. "
           "Without seaborn, standard matplotlib plots will not look very charming. "
           "It's recommended to install it via pip install seaborn")


def read_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--searcher', type=str, default='tpe', choices=['tpe', 'rand'])
    parser.add_argument('--max_evals', type=int, default=100)
    parser.add_argument('--train', type=str, required=True, help="training set")
    parser.add_argument('--holdout', type=str, required=True, help="holdout set")
    parser.add_argument('--vw_space', type=str, required=True, help="hyperopt search space (must be 'quoted')")
    parser.add_argument('--outer_loss_function', default='logistic',
                        choices=['logistic', 'roc-auc'])  # TODO: implement squared, hinge, quantile, PR-auc
    parser.add_argument('--regression', action='store_true', default=False, help="""regression (continuous class labels)
                                                                        or classification (-1 or 1, default value).""")
    parser.add_argument('--plot', action='store_true', default=False, help=("Plot the results in the end. "
                                                                            "Requires matplotlib and "
                                                                            "(optionally) seaborn to be installed."))
    parser.add_argument('--stepwise', type=str, metavar='ALL_NAMESPACES+INITIAL_NAMESPACES',
                        help="Turns on bidirectional stepwise feature selection.")
    args = parser.parse_args()
    return args


class HyperoptSpaceConstructor(object):
    """
    Takes command-line input and transforms it into hyperopt search space
    An example of command-line input:

    --algorithms=ftrl,sgd --l2=1e-8..1e-4~LO -l=0.01..10~L --ftrl_beta=0.01..1 --passes=1..10~I -q=SE+SZ+DR,SE~O
    """

    def __init__(self, command):
        self.command = command
        self.space = None
        self.algorithm_metadata = {
            'ftrl': {'arg': '--ftrl', 'prohibited_flags': set()},
            'sgd': {'arg': '', 'prohibited_flags': {'--ftrl_alpha', '--ftrl_beta'}}
        }

        self.range_pattern = re.compile("[^~]+")  # re.compile("(?<=\[).+(?=\])")
        self.distr_pattern = re.compile("(?<=~)[IOL]*")  # re.compile("(?<=\])[IOL]*")
        self.only_continuous = re.compile("(?<=~)[IL]*")  # re.compile("(?<=\])[IL]*")

    def _process_vw_argument(self, arg, value, algorithm):
        try:
            distr_part = self.distr_pattern.findall(value)[0]
        except IndexError:
            distr_part = ''
        range_part = self.range_pattern.findall(value)[0]
        is_continuous = '..' in range_part
        is_quadratic = arg.startswith('-q#')

        ocd = self.only_continuous.findall(value)
        if not is_continuous and len(ocd)> 0 and ocd[0] != '':
            raise ValueError(("Need a range instead of a list of discrete values to define "
                              "uniform or log-uniform distribution. "
                              "Please, use [min..max]%s form") % (distr_part))

        if is_continuous and is_quadratic:
            raise ValueError(("You must directly specify namespaces for quadratic features "
                              "as a list of values, not as a parametric distribution"))

        hp_choice_name = "_".join([algorithm, arg.replace('-', '')])

        try_omit_zero = 'O' in distr_part
        distr_part = distr_part.replace('O', '')

        if is_continuous:
            vmin, vmax = [float(i) for i in range_part.split('..')]

            if distr_part == 'L':
                distrib = hp.loguniform(hp_choice_name, log(vmin), log(vmax))
            elif distr_part == '':
                distrib = hp.uniform(hp_choice_name, vmin, vmax)
            elif distr_part == 'I':
                distrib = hp.quniform(hp_choice_name, vmin, vmax, 1)
            elif distr_part in {'LI', 'IL'}:
                distrib = hp.qloguniform(hp_choice_name, log(vmin), log(vmax), 1)
            else:
                raise ValueError("Cannot recognize distribution: %s" % (distr_part))
        else:
            possible_values = range_part.split(',')
            if is_quadratic:
                possible_values = [v.replace('+', ' -q ') for v in possible_values]
            distrib = hp.choice(hp_choice_name, possible_values)

        if try_omit_zero:
            hp_choice_name_outer = hp_choice_name + '_outer'
            distrib = hp.choice(hp_choice_name_outer, ['omit', distrib])

        return distrib

    def string_to_pyll(self):
        line = shlex.split(self.command)

        algorithms = ['sgd']
        for arg in line:
            arg, value = arg.split('=')
            if arg == '--algorithms':
                algorithms = set(self.range_pattern.findall(value)[0].split(','))
                if tuple(self.distr_pattern.findall(value)) not in {(), ('O',)}:
                    raise ValueError(("Distribution options are prohibited for --algorithms flag. "
                                      "Simply list the algorithms instead (like --algorithms=ftrl,sgd)"))
                elif self.distr_pattern.findall(value) == ['O']:
                    algorithms.add('sgd')

                for algo in algorithms:
                    if algo not in self.algorithm_metadata:
                        raise NotImplementedError(("%s algorithm is not found. "
                                                   "Supported algorithms by now are %s")
                                                  % (algo, str(self.algorithm_metadata.keys())))
                break

        self.space = {algo: {'type': algo, 'argument': self.algorithm_metadata[algo]['arg']} for algo in algorithms}
        count = 0
        for algo in algorithms:
            for arg in line:
                count += 1
                arg, value = arg.split('=')
                if arg == '--algorithms':
                    continue
                if arg not in self.algorithm_metadata[algo]['prohibited_flags']:
                    arg = '%s#%i' % (arg, count)
                    distrib = self._process_vw_argument(arg, value, algo)
                    self.space[algo][arg] = distrib
                else:
                    pass
        self.space = hp.choice('algorithm', self.space.values())
        #print 'Space: ', self.space

class HyperOptimizer(object):
    def __init__(self, train_set, holdout_set, command, max_evals=100,
                 outer_loss_function='logistic',
                 searcher='tpe', is_regression=False, stepwise=None):
        self.train_set = train_set
        self.holdout_set = holdout_set

        self.train_model = './current.model'
        self.holdout_pred = './holdout.pred'
        self.trials_output = './trials.json'
        self.hyperopt_progress_plot = './hyperopt_progress.png'
        self.log = './log.log'

        self.logger = self._configure_logger()

        # hyperopt parameter sample, converted into a string with flags
        self.param_suffix = None
        self.validation_param_suffix = None
        self.stepwise_param_suffix = ''
        self.train_command = None
        self.validate_command = None

        self.y_true_train = []
        self.y_true_holdout = []

        self.outer_loss_function = outer_loss_function
        self.space = self._get_space(command)
        self.max_evals = max_evals
        self.searcher = searcher
        self.is_regression = is_regression

        self.hyperopt_best_loss = None
        self.hyperopt_best_train_command = None
        self.hyperopt_best_param_suffix = None
        self.hyperopt_best_validation_param_suffix = None

        self.is_stepwise = stepwise is not None
        if self.is_stepwise:
            self.trials_output_dir = './trials'
            namespaces = stepwise.split('+')
            if len(namespaces) > 2:
                raise ValueError("You can not specify more than one plus sign in --stepwise")
            if namespaces[0] == '':
                raise ValueError("ALL_NAMESPACES can not be empty in --stepwise")
            self.all_namespaces = set(namespaces[0])
            self.current_namespaces = set(namespaces[1])
            if not self.current_namespaces.issubset(self.all_namespaces):
                raise ValueError("INITIAL_NAMESPACES must be a subset of ALL_NAMESPACES in --stepwise")
            self.stepwise_path = ''.join(sorted(self.current_namespaces))
            self.stepwise_best_loss = None
            self.stepwise_best_train_command = None
            self.stepwise_best_namespaces = None
            self.stepwise_best_path = None
            self.current_step = 0

        self.cache = {}
        self.trials = None
        self.current_trial = None
        self.total_best_loss = None

    def _get_space(self, command):
        hs = HyperoptSpaceConstructor(command)
        hs.string_to_pyll()
        return hs.space

    def _configure_logger(self):
        LOGGER_FORMAT = "%(asctime)s,%(msecs)03d %(levelname)-8s [%(name)s/%(module)s:%(lineno)d]: %(message)s"
        LOGGER_DATEFMT = "%Y-%m-%d %H:%M:%S"
        LOGFILE = self.log

        logging.basicConfig(format=LOGGER_FORMAT,
                            datefmt=LOGGER_DATEFMT,
                            level=logging.DEBUG)
        formatter = logging.Formatter(LOGGER_FORMAT, datefmt=LOGGER_DATEFMT)

        file_handler = logging.FileHandler(LOGFILE)
        file_handler.setFormatter(formatter)

        logger = logging.getLogger()
        logger.addHandler(file_handler)
        return logger

    def compose_hyperopt_param_suffix(self, **kwargs):
        #print 'KWARGS: ', kwargs
        args = []
        validation_args = []
        for key in sorted(kwargs.keys()):
            value = kwargs[key]
            key = key.split('#')[0]
            if key.startswith('-') and value != 'omit':
                if key in ['--passes']: #, '--rank', '--lrq']:
                    value = int(value)
                value = str(value)
                args.append(key)
                args.append(value)
                if key in ['--keep', '--ignore']:
                    validation_args.append(key)
                    validation_args.append(value)
        self.param_suffix = ' '.join(args) + ' ' + (kwargs['argument'])
        self.validation_param_suffix = ' '.join(validation_args)

    def compose_stepwise_param_suffix(self):
        self.stepwise_param_suffix = '--keep ' + ''.join(sorted(self.current_namespaces)) \
            if len(self.current_namespaces) > 0 \
            else ''

    def compose_vw_train_command(self):
        data_part = ('vw -d %s -f %s --holdout_off -c '
                     % (self.train_set, self.train_model))
        self.train_command = ' '.join([data_part, self.param_suffix, self.stepwise_param_suffix])

    def compose_vw_validate_command(self):
        data_part = 'vw -t -d %s -i %s -p %s --holdout_off -c' \
                    % (self.holdout_set, self.train_model, self.holdout_pred)
        self.validate_command = ' '.join([data_part, self.validation_param_suffix, self.stepwise_param_suffix])

    def fit_vw(self):
        self.logger.info("executing the following command (training): %s" % self.train_command)
        subprocess.call(shlex.split(self.train_command))

    def validate_vw(self):
        self.logger.info("executing the following command (validation): %s" % self.validate_command)
        subprocess.call(shlex.split(self.validate_command))

    def get_y_true_train(self):
        self.logger.info("loading true train class labels...")
        yh = open(self.train_set, 'r')
        self.y_true_train = []
        for line in yh:
            self.y_true_train.append(int(line.strip()[0:2]))
        if not self.is_regression:
            self.y_true_train = [(i + 1.) / 2 for i in self.y_true_train]
        self.logger.info("train length: %d" % len(self.y_true_train))

    def get_y_true_holdout(self):
        self.logger.info("loading true holdout class labels...")
        yh = open(self.holdout_set, 'r')
        self.y_true_holdout = []
        for line in yh:
            self.y_true_holdout.append(int(line.strip()[0:2]))
        if not self.is_regression:
            self.y_true_holdout = [(i + 1.) / 2 for i in self.y_true_holdout]
        self.logger.info("holdout length: %d" % len(self.y_true_holdout))

    def validation_metric_vw(self):
        v = open('%s' % self.holdout_pred, 'r')
        y_pred_holdout = []
        for line in v:
            y_pred_holdout.append(float(line.strip().split(' ')[0]))

        if self.outer_loss_function == 'logistic':
            y_pred_holdout_proba = [1. / (1 + exp(-i)) for i in y_pred_holdout]
            loss = log_loss(self.y_true_holdout, y_pred_holdout_proba)

        elif self.outer_loss_function == 'squared':  # TODO: write it
            pass

        elif self.outer_loss_function == 'hinge':  # TODO: write it
            pass

        elif self.outer_loss_function == 'roc-auc':
            y_pred_holdout_proba = [1. / (1 + exp(-i)) for i in y_pred_holdout]
            fpr, tpr, _ = roc_curve(self.y_true_holdout, y_pred_holdout_proba)
            loss = -auc(fpr, tpr)

        self.logger.info('parameter suffix: %s' % ' '.join([self.param_suffix, self.stepwise_param_suffix]))
        self.logger.info('loss value: %.6f' % loss)

        return loss

    def run_trial(self):
        start = dt.now()
        self.current_trial += 1
        message = '\n\nStarting trial no.%d' % self.current_trial
        if self.is_stepwise:
            message += '\nstepwise status: step no.%d, %d namespace(s), %s' \
                % (self.current_step, len(self.current_namespaces), self.stepwise_path)
            if self.total_best_loss is not None:
                message += '\nbest loss so far (not considering current trials): %.6f' % self.total_best_loss
        self.logger.info(message)

        self.compose_vw_train_command()
        self.compose_vw_validate_command()

        if self.train_command in self.cache:
            loss = self.cache[self.train_command]
            self.logger.info('found cached loss value: %.6f' % loss)
        else:
            self.fit_vw()
            self.validate_vw()
            loss = self.validation_metric_vw()
            self.cache[self.train_command] = loss
            os.remove(self.train_model)
            os.remove(self.holdout_pred)

        finish = dt.now()
        elapsed = finish - start
        self.logger.info("evaluation time for this step: %s" % str(elapsed))

        return {
            'status': STATUS_OK,
            'loss': loss,  # TODO: include also train loss tracking in order to prevent overfitting
            'eval_time': elapsed.seconds,
            'train_command': self.train_command,
            'current_trial': self.current_trial,
            }

    def optimize(self, parallel=False):  # TODO: implement parallel search with MongoTrials
        def hyperopt_trial(kwargs):
            self.compose_hyperopt_param_suffix(**kwargs)
            result = self.run_trial()
            if self.current_trial == 1 or result['loss'] < self.hyperopt_best_loss:
                self.hyperopt_best_loss = result['loss']
                self.hyperopt_best_train_command = self.train_command
                self.hyperopt_best_param_suffix = self.param_suffix
                self.hyperopt_best_validation_param_suffix = self.validation_param_suffix
            return result

        def stepwise_trial(results):
            self.compose_stepwise_param_suffix()
            result = self.run_trial()
            if self.current_trial == 1 or result['loss'] < self.stepwise_best_loss:
                self.stepwise_best_loss = result['loss']
                self.stepwise_best_train_command = self.train_command
                self.stepwise_best_namespaces = self.current_namespaces.copy()
                self.stepwise_best_path = self.stepwise_path
            self.logger.info("Stepwise feature selection completed %d trials with best loss: %.6f" % (self.current_trial, self.stepwise_best_loss))
            results.append(result)

        def search_namespaces(results):
            path = self.stepwise_path
            if len(self.current_namespaces) > 1:
                for namespace in self.current_namespaces.copy():
                    self.current_namespaces.remove(namespace)
                    self.stepwise_path = ''.join([path, '-', namespace, '*'])
                    stepwise_trial(results)
                    self.current_namespaces.add(namespace)
            if len(self.current_namespaces) < len(self.all_namespaces):
                for namespace in self.all_namespaces - self.current_namespaces:
                    self.current_namespaces.add(namespace)
                    self.stepwise_path = ''.join([path, '+', namespace, '*'])
                    stepwise_trial(results)
                    self.current_namespaces.remove(namespace)
            self.stepwise_path = path

        if self.searcher == 'tpe':
            algo = tpe.suggest
        elif self.searcher == 'rand':
            algo = rand.suggest

        logging.debug("starting hyperparameter optimization" 
            + (" with bidirectional stepwise feature selection" if self.is_stepwise else "") + "...")

        if self.is_stepwise:
            os.makedirs(self.trials_output_dir)

        while True:
            if self.is_stepwise:
                self.current_step += 1
                self.trials_output = '%s/%04d-hyperopt.json' % (self.trials_output_dir, self.current_step)
                self.compose_stepwise_param_suffix()

            self.trials = Trials()
            self.current_trial = 0
            best_params = fmin(hyperopt_trial, space=self.space, trials=self.trials, algo=algo, max_evals=self.max_evals)
            self.logger.debug("the best hyperopt parameters: %s" % str(best_params))

            json.dump(self.trials.results, open(self.trials_output, 'w'))
            self.logger.info('All the trials results are saved at %s' % self.trials_output)

            self.logger.info("\n\nA full training command with the best hyperparameters: \n%s\n\n" % self.hyperopt_best_train_command)
            self.logger.info("\n\nThe best holdout loss value: \n%s\n\n" % self.hyperopt_best_loss)

            if not self.is_stepwise:
                break

            if self.current_step == 1 or self.hyperopt_best_loss < self.total_best_loss:
                self.total_best_loss = self.hyperopt_best_loss
                total_best_train_command = self.hyperopt_best_train_command
                total_best_param_suffix = self.hyperopt_best_param_suffix
                total_best_validation_param_suffix = self.hyperopt_best_validation_param_suffix
            else:
                self.logger.info("Hyperopt didn't found better parameters on this step! Using previous ones")

            self.trials_output = '%s/%04d-stepwise.json' % (self.trials_output_dir, self.current_step)
            self.param_suffix = total_best_param_suffix
            self.validation_param_suffix = total_best_validation_param_suffix

            results = []
            self.current_trial = 0
            search_namespaces(results)
            self.logger.debug("the best namespaces: %s" % ''.join(sorted(self.stepwise_best_namespaces)))

            json.dump(results, open(self.trials_output, 'w'))
            self.logger.info('All the trials results are saved at %s' % self.trials_output)

            self.logger.info("\n\nA full training command with the best hyperparameters: \n%s\n\n" % self.stepwise_best_train_command)
            self.logger.info("\n\nThe best holdout loss value: \n%s\n\n" % self.stepwise_best_loss)

            if self.stepwise_best_loss <= self.hyperopt_best_loss:
                self.total_best_loss = self.stepwise_best_loss
                total_best_train_command = self.stepwise_best_train_command
                self.current_namespaces = self.stepwise_best_namespaces.copy()
                self.stepwise_path = self.stepwise_best_path.rstrip('*')
            else:
                break

        if self.is_stepwise:
            self.logger.info("\n\nStepwise feature selection completed\n\n")
            self.logger.info("\n\nFeature selection path: %s\n\n" % self.stepwise_path)
            self.logger.info("\n\nA full training command with the best hyperparameters: \n%s\n\n" % total_best_train_command)
            self.logger.info("\n\nThe best holdout loss value: \n%s\n\n" % self.total_best_loss)

    def plot_progress(self):
        try:
            sns.set_palette('Set2')
            sns.set_style("darkgrid", {"axes.facecolor": ".95"})
        except:
            pass

        self.logger.debug('plotting...')
        plt.figure(figsize=(15,10))
        plt.subplot(211)
        plt.plot(self.trials.losses(), '.', markersize=12)
        plt.title('Per-Iteration Outer Loss', fontsize=16)
        plt.ylabel('Outer loss function value')
        if self.outer_loss_function in ['logloss']:
            plt.yscale('log')
        xticks = [int(i) for i in np.linspace(plt.xlim()[0], plt.xlim()[1], min(len(self.trials.losses()), 11))]
        plt.xticks(xticks, xticks)


        plt.subplot(212)
        plt.plot(np.minimum.accumulate(self.trials.losses()), '.', markersize=12)
        plt.title('Cumulative Minimum Outer Loss', fontsize=16)
        plt.xlabel('Iteration number')
        plt.ylabel('Outer loss function value')
        xticks = [int(i) for i in np.linspace(plt.xlim()[0], plt.xlim()[1], min(len(self.trials.losses()), 11))]
        plt.xticks(xticks, xticks)

        plt.tight_layout()
        plt.savefig(self.hyperopt_progress_plot)
        self.logger.info('The diagnostic hyperopt progress plot is saved: %s' % self.hyperopt_progress_plot)


def main():
    args = read_arguments()
    h = HyperOptimizer(train_set=args.train, holdout_set=args.holdout, command=args.vw_space,
                       max_evals=args.max_evals,
                       outer_loss_function=args.outer_loss_function,
                       searcher=args.searcher, is_regression=args.regression,
                       stepwise = args.stepwise)
    h.get_y_true_holdout()
    h.optimize()
    if args.plot:
        h.plot_progress()


if __name__ == '__main__':
    main()
