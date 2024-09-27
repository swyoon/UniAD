"""
Extract EfficientNet features of images.

Usage:
cd experiments/MVTec-AD
bash extract_representation.sh
"""
import argparse
import logging
import os
import pprint
import shutil
import time
import sys
sys.path.append('.')

import torch
import torch.distributed as dist
import torch.optim
import yaml
from datasets.data_builder import build_dataloader
from easydict import EasyDict
from models.model_helper import ModelHelper
from tensorboardX import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP
from utils.criterion_helper import build_criterion
from utils.dist_helper import setup_distributed
from utils.eval_helper import dump, log_metrics, merge_together, performances
from utils.lr_helper import get_scheduler
from utils.misc_helper import (
    AverageMeter,
    create_logger,
    get_current_time,
    load_state,
    save_checkpoint,
    set_random_seed,
    update_config,
)
from utils.optimizer_helper import get_optimizer
from utils.vis_helper import visualize_compound, visualize_single

parser = argparse.ArgumentParser(description="UniAD Framework")
parser.add_argument("--config", default="./config.yaml")
parser.add_argument("-e", "--evaluate", action="store_true")
# parser.add_argument("--local_rank", default=None, help="local rank for dist")


def main():
    global args, config, key_metric, best_metric
    args = parser.parse_args()

    with open(args.config) as f:
        config = EasyDict(yaml.load(f, Loader=yaml.FullLoader))

    config.port = config.get("port", None)
    rank, world_size = setup_distributed(port=config.port)
    config = update_config(config)

    config.exp_path = os.path.dirname(args.config)
    config.save_path = os.path.join(config.exp_path, config.saver.save_dir)
    config.log_path = os.path.join(config.exp_path, config.saver.log_dir)
    config.evaluator.eval_dir = os.path.join(config.exp_path, config.evaluator.save_dir)
    if rank == 0:
        os.makedirs(config.save_path, exist_ok=True)
        os.makedirs(config.log_path, exist_ok=True)

        current_time = get_current_time()
        tb_logger = SummaryWriter(config.log_path + "/events_dec/" + current_time)
        logger = create_logger(
            "global_logger", config.log_path + "/dec_{}.log".format(current_time)
        )
        logger.info("args: {}".format(pprint.pformat(args)))
        logger.info("config: {}".format(pprint.pformat(config)))
    else:
        tb_logger = None

    random_seed = config.get("random_seed", None)
    reproduce = config.get("reproduce", None)
    if random_seed:
        set_random_seed(random_seed, reproduce)

    # create model
    model = ModelHelper(config.net)
    model.cuda()
    local_rank = int(os.environ["LOCAL_RANK"])
    model = DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=True,
    )

    layers = []
    for module in config.net:
        layers.append(module["name"])
    frozen_layers = config.get("frozen_layers", [])
    active_layers = list(set(layers) ^ set(frozen_layers))
    if rank == 0:
        logger.info("layers: {}".format(layers))
        logger.info("active layers: {}".format(active_layers))

    # parameters needed to be updated
    parameters = [
        {"params": getattr(model.module, layer).parameters()} for layer in active_layers
    ]

    optimizer = get_optimizer(parameters, config.trainer.optimizer)
    lr_scheduler = get_scheduler(optimizer, config.trainer.lr_scheduler)

    key_metric = config.evaluator["key_metric"]
    best_metric = 0
    last_epoch = 0

    # load model: auto_resume > resume_model > load_path
    auto_resume = config.saver.get("auto_resume", True)
    resume_model = config.saver.get("resume_model", None)
    load_path = config.saver.get("load_path", None)

    if resume_model and not resume_model.startswith("/"):
        resume_model = os.path.join(config.exp_path, resume_model)
    lastest_model = os.path.join(config.save_path, "ckpt.pth.tar")
    if auto_resume and os.path.exists(lastest_model):
        resume_model = lastest_model
    if resume_model:
        best_metric, last_epoch = load_state(resume_model, model, optimizer=optimizer)
    elif load_path:
        if not load_path.startswith("/"):
            load_path = os.path.join(config.exp_path, load_path)
        load_state(load_path, model)

    train_loader, val_loader = build_dataloader(config.dataset, distributed=False)

    save_data(
        train_loader,
        'train_data.pkl',
        model,
    )
    save_data(
        # train_loader,
        val_loader,
        'val_data.pkl',
        model,
    )

def save_data(
    dataloader,
    out_path,
    model,
):
    model.eval()

    end = time.time()

    l = []
    from tqdm import tqdm
    for i, input in enumerate(tqdm(dataloader)):

        # forward
        with torch.no_grad():
            outputs = model(input)
        # data_to_save = {k: outputs[k].cpu() if hasattr(outputs[k], 'cpu') else outputs[k] for k in ['mask', 'features', 'strides', 'feature_align',  'label', 'clsname']}
        data_to_save = {k: outputs[k].cpu() if hasattr(outputs[k], 'cpu') else outputs[k] for k in ['mask', 'feature_align',  'label', 'clsname']}
        l.append(data_to_save)
        continue

    from torch.utils.data._utils.collate import default_collate
    import itertools
    keys = l[0].keys()
    out = {}
    for k in keys:
        ll = [x[k] for x in l]
        if isinstance(ll[0], list):
            ll = list(itertools.chain(*ll))
            print(k, len(ll))
        elif isinstance(ll[0], torch.Tensor):
            ll = torch.cat(ll, dim=0)
            print(k, ll.shape)
        out[k] = ll
    # torch.save(out, 'train_data.pkl')
    torch.save(out, out_path)
    # import sys
    # sys.exit()


def validate(val_loader, model):
    batch_time = AverageMeter(0)
    losses = AverageMeter(0)

    model.eval()
    rank = dist.get_rank()
    logger = logging.getLogger("global_logger")
    criterion = build_criterion(config.criterion)
    end = time.time()

    if rank == 0:
        os.makedirs(config.evaluator.eval_dir, exist_ok=True)
    # all threads write to config.evaluator.eval_dir, it must be made before every thread begin to write
    dist.barrier()

    with torch.no_grad():
        for i, input in enumerate(val_loader):
            # forward
            outputs = model(input)
            dump(config.evaluator.eval_dir, outputs)

            # record loss
            loss = 0
            for name, criterion_loss in criterion.items():
                weight = criterion_loss.weight
                loss += weight * criterion_loss(outputs)
            num = len(outputs["filename"])
            losses.update(loss.item(), num)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % config.trainer.print_freq_step == 0 and rank == 0:
                logger.info(
                    "Test: [{0}/{1}]\tTime {batch_time.val:.3f} ({batch_time.avg:.3f})".format(
                        i + 1, len(val_loader), batch_time=batch_time
                    )
                )

    # gather final results
    dist.barrier()
    total_num = torch.Tensor([losses.count]).cuda()
    loss_sum = torch.Tensor([losses.avg * losses.count]).cuda()
    dist.all_reduce(total_num, async_op=True)
    dist.all_reduce(loss_sum, async_op=True)
    final_loss = loss_sum.item() / total_num.item()

    ret_metrics = {}  # only ret_metrics on rank0 is not empty
    if rank == 0:
        logger.info("Gathering final results ...")
        # total loss
        logger.info(" * Loss {:.5f}\ttotal_num={}".format(final_loss, total_num.item()))
        fileinfos, preds, masks = merge_together(config.evaluator.eval_dir)
        shutil.rmtree(config.evaluator.eval_dir)
        # evaluate, log & vis
        ret_metrics = performances(fileinfos, preds, masks, config.evaluator.metrics)
        log_metrics(ret_metrics, config.evaluator.metrics)
        if args.evaluate and config.evaluator.get("vis_compound", None):
            visualize_compound(
                fileinfos,
                preds,
                masks,
                config.evaluator.vis_compound,
                config.dataset.image_reader,
            )
        if args.evaluate and config.evaluator.get("vis_single", None):
            visualize_single(
                fileinfos,
                preds,
                config.evaluator.vis_single,
                config.dataset.image_reader,
            )
    model.train()
    return ret_metrics


if __name__ == "__main__":
    main()
