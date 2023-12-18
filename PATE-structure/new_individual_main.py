# coding=UTF8
from os import makedirs
from pathlib import Path
import shutil
import click
from loguru import logger
from billiard.process import Process

from data_now.data_factory import DataFactory
# 替换前：
# from multiprocessing import Pool
# #替换后：
# from billiard import Pool

from data_now.parameters import ExperimentPlan, ExperimentParameters
from data_now import train_students, run_votings, train_teachers
import warnings

warnings.simplefilter('ignore', category=UserWarning)
"""
This file executes the complete PATE pipeline on for each parameter combination that is specified.
First, the dataset is divided into private, public, and test parts. Then, the private subset is allocated to the
teachers depending on the privacy personalization and the collector. The teachers are trained on their corresponding
subsets and stored. They are loaded afterwards for the voting that produces labels for the public subset which are
stored then. Thereafter, the labels are loaded together with the public subset to train a student. For the teachers'
and student's training as well as for the voting, statistics are stored in three different result files for subsequent
analysis.
"""
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def run_pate(params_path, data_dir, out_dir, train_dir, data_name, image_size, num_classes, log_file=None, n_reduce_teachers=0):
    """
    params_path:配置文件地址
    data_dir:x.npy,y.npy文件保存地址
    out_dir:结果输出地址
    train_dir:划分好的train数据地址
    test_dir:划分好的test数据地址
    data_name:训练数据的名字
    """
    if log_file:
        logger.remove()
        logger.add(log_file)
    # 加载训练参数
    plan = ExperimentPlan.load(params_path=Path(params_path),
                               data_dir=Path(data_dir),
                               out_dir=Path(out_dir))

    logger.info("Starting study:")
    logger.info(plan)
    # 加载数据集
    data_factory = DataFactory(data_name=plan.data.data_name,
                               data_dir=plan.resources.data_dir,
                               out_dir=plan.resources.out_dir, train_dir=train_dir)

    logger.info("Preparing data:")
    # 划分数据集
    data_factory.write_splits(seeds=plan.pate.seeds,
                              n_test=plan.data.n_test,
                              n_public=plan.data.n_public,
                              n_private=plan.data.n_private,
                              n_valid=plan.data.n_valid,
                              num_classes=num_classes)

    prms: ExperimentParameters
    '''每一个'''
    for prms in plan.derive_experiment_parameters(data_factory=data_factory):
        logger.info(f"Starting experiment:")
        logger.info(prms)

        logger.info(f"Training teachers:")
        # 训练教师机
        budgets_per_sample, mapping_t2p = train_teachers.main(prms,
                                                              data_factory,
                                                              n_reduce_teachers=n_reduce_teachers)

        run_votings.main(
            prms,
            data_factory,
            num_classes=num_classes,
            data_name=data_name, budgets_per_sample=budgets_per_sample, mapping_t2p=mapping_t2p
        )
        logger.info(f"Training students:")
        train_students.main(
            prms,
            data_factory, data_name, image_size
        )


def main(params_path=None, data_dir=r"G:\RS1130\data\EuroSAT\2750", data_name='EuroSAT', image_size=64, out_dir='', train_dir=r"G:\RS1130\data\EuroSAT\2750", num_classes=10):
    """
    params_path:配置文件地址
    data_dir:x.npy,y.npy文件保存地址
    out_dir:结果输出地址
    train_dir:划分好的train数据地址
    test_dir:划分好的test数据地址
    data_name:训练数据的名字
    """
    # params_path, data_dir, out_dir, train_dir, data_name, image_size, num_classes, log_file=None, n_reduce_teachers=0
    run_pate(params_path=params_path, data_dir=data_dir, data_name=data_name, num_classes=num_classes, image_size=image_size, out_dir=out_dir, train_dir=train_dir, log_file=None,
             n_reduce_teachers=0)


@click.command()
@click.option('--set_params_dir',
              '-s',
              required=True,
              type=click.Path(file_okay=True))
@click.option('--data_dir',
              '-d',
              required=True,
              type=click.Path(dir_okay=True))
@click.option('--out_root_dir',
              '-o',
              required=True,
              type=click.Path(dir_okay=True))
def run_experiment_set(set_params_dir, data_dir, out_root_dir):
    set_params_dir = Path(set_params_dir).resolve()
    out_root_dir = Path(out_root_dir).resolve()
    params_paths = [
        f.resolve() for f in Path(set_params_dir).iterdir() if f.is_file()
    ]

    experiment_set_name = set_params_dir.name

    # iterate over experiment plans of set
    # and create a process for each plan
    processes = []
    for params_path in params_paths:
        experiment_plan_name = params_path.name.split('.')[0]
        # create subfolder for exeriment plan
        out_subdir = out_root_dir / experiment_plan_name
        makedirs(out_subdir, exist_ok=True)

        # copy parameter file for experiment plan
        shutil.copy(params_path, out_subdir)

        # path to logfile
        log_file = out_subdir / 'log.txt'

        # call pate
        process = Process(
            target=run_pate,
            kwargs={
                'params_path': params_path,
                'data_dir': data_dir,
                'out_dir': out_subdir,
                'log_file': log_file,
                'n_reduce_teachers': 0,
            },
            name=f'per-point-pate:{experiment_set_name}:{experiment_plan_name}'
            , daemon=False)
        processes.append(process)
        process.start()


if __name__ == "__main__":
    # 参数文件.yaml
    # params_path = r"G:\RS1130\codeNew\individualized-pate-main\individualized-pate-main\45.yaml"
    # # 数据集输入存储x.npy y.npy文件
    # data_dir = r"G:\RS1130\data\NWPU-RESISC45\NWPU-RESISC45"
    # # 结果输出
    # data_name = "NWPU-RESISC45"
    # out_dir = r"G:\RS1130\data\45_result_resnet18_4"
    # train_dir = r"G:\RS1130\data\NWPU-RESISC45\split_NR45\train"
    # test_dir = r"G:\RS1130\data\NWPU-RESISC45\split_NR45\test"
    # # # 主入口
    # main(params_path, data_dir, out_dir, train_dir, test_dir, data_name)
    # params_path = r"G:\RS1130\codeNew\individualized-pate-main\individualized-pate-main\experiment_plans\templates\cifar.yaml"
    # # # 数据集输入
    # data_dir = r"G:\RS1130\data\cifar"
    # # 结果输出
    # out_dir = r"G:\RS1130\data\cifar-result"
    # data_name="cifar"
    # # 主入口
    # main(params_path, data_dir, out_dir,"","",data_name)
    #     G:\RS1130\data\EuroSAT\2750
    # main()
    # params_path = r"G:\RS1130\codeNew\individualized-pate-SKAL\PATE-structure\10-personalized-t25.yaml"
    # # 数据集dir_path
    # data_dir = r"G:\RS1130\data\EuroSAT\2750"
    # # 结果输出
    # out_dir = r"G:\RS1130\data\10_skal-flow-tttttt"
    # # 划分好的数据存储
    # train_dir = r"G:\RS1130\data\EuroSAT\2750"
    # image_size = 64
    # # test_dir=r"G:\RS1130\data\EuroSAT\split_data\valid"
    # # 主入口
    # # main(params_path, data_dir, out_dir, train_dir, data_name="EuroSAT", image_size=image_size)
    # num_classes = 10
    # main(params_path, data_dir, "EuroSAT", image_size, out_dir, train_dir, num_classes)
    # main(params_path = r"G:\RS1130\codeNew\individualized-pate-SKAL\PATE-structure\10-personalized-t100.yaml",out_dir = r"G:\RS1130\data\10_skal-flow-t100")
    # main(params_path=r"G:\RS1130\codeNew\individualized-pate-SKAL\PATE-structure\10-personalized-t250.yaml", out_dir=r"G:\RS1130\data\10_skal-flow-t250")
    # main(params_path=r"G:\RS1130\codeNew\individualized-pate-SKAL\PATE-structure\10-personalized-t300.yaml", out_dir=r"G:\RS1130\data\10_skal-flow-t300")
    main(params_path=r"G:\RS1130\codeNew\individualized-pate-SKAL\PATE-structure\10-personalized-t50.yaml", out_dir=r"G:\RS1130\data\10_skal-flow-t50")
    # main(params_path=r"G:\RS1130\codeNew\individualized-pate-SKAL\PATE-structure\10-personalized-t75.yaml", out_dir=r"G:\RS1130\data\10_skal-flow-t75")

    # main(params_path=r"G:\RS1130\codeNew\individualized-pate-SKAL\PATE-structure\10-personalized-t125.yaml", out_dir=r"G:\RS1130\data\10_skal-flow-t125")
    #
    # main(params_path=r"G:\RS1130\codeNew\individualized-pate-SKAL\PATE-structure\10-personalized-t150.yaml", out_dir=r"G:\RS1130\data\10_skal-flow-t150")
    #
    # main(params_path=r"G:\RS1130\codeNew\individualized-pate-SKAL\PATE-structure\10-personalized-t175.yaml", out_dir=r"G:\RS1130\data\10_skal-flow-t175")
    #
    # main(params_path=r"G:\RS1130\codeNew\individualized-pate-SKAL\PATE-structure\10-personalized-t200.yaml", out_dir=r"G:\RS1130\data\10_skal-flow-t200")
