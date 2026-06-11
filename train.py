import ever as er
import torch
import numpy as np
import os
from data.EarthMiss import COLOR_MAP
from tqdm import tqdm
import random
import torch.distributed as dist
from module.slide_test import slide_inference
from module.viz import VisualizeSegmm
from module.baseline.MetaRS import MetaRS
from ever import registry
from ever.interface.callback import SaveCheckpointCallback

# Disk-saving policy: the framework normally writes a full ~1.3 GB checkpoint
# every `save_ckpt_interval_epoch` epochs and once more after training. We
# disable that automatic saving and instead keep ONLY the single best-by-mIoU
# checkpoint (overwritten in place as Best.pth) from the evaluation hook below.
SaveCheckpointCallback.func = lambda self: None

# Optional per-tile prediction PNGs during training-eval are off by default to
# avoid filling the disk; set VIZ_DURING_TRAIN=1 to re-enable.
_VIZ = os.environ.get("VIZ_DURING_TRAIN", "0") == "1"


def _miou_from_table(table):
    """Extract the mean-IoU scalar from a PixelMetric AccTable."""
    for row in table._rows:
        cells = [str(c) for c in row]
        if "mean" in cells:
            return float(row[cells.index("mean") + 1])
    return None


def evaluate_cls_fn(self, test_dataloader, config=None):
    self.model.eval()
    classes = self.model.module.config.num_classes if self.model.module.config.num_classes != 1 else 2
    metric_op = er.metric.PixelMetric(classes, logdir=self._model_dir, logger=self.logger, class_names=list(COLOR_MAP.keys()))

    if _VIZ:
        vis_dir = os.path.join(self._model_dir, 'vis-{}'.format(self.checkpoint.global_step))
        palette = np.array(list(COLOR_MAP.values())).reshape(-1).tolist()
        viz_op = VisualizeSegmm(vis_dir, palette)

    with torch.no_grad():
        for img, gt in tqdm(test_dataloader):
            img = img.to(torch.device('cuda'))
            y_true = gt['cls']
            y_true = y_true.cpu()
            if config is not None and config.get('slide_test', False):
                pred = slide_inference(self.model, img, gt, config['slide_test'])
            else:
                pred = self.model(img, gt)

            y_pred = pred.argmax(dim=1).cpu()

            valid_inds = y_true != -1
            metric_op.forward(y_true[valid_inds], y_pred[valid_inds])

            if _VIZ:
                for clsmap, imname in zip(y_pred, gt['fname']):
                    if "tif" in imname:
                        viz_op(clsmap.cpu().numpy().astype(np.uint8), imname.replace('tif', 'png'))
                    elif "jpg" in imname:
                        viz_op(clsmap.cpu().numpy().astype(np.uint8), imname.replace('jpg', 'png'))
                    else:
                        viz_op(clsmap.cpu().numpy().astype(np.uint8), imname)

    table = metric_op.summary_all()  # all_gather happens here -> every rank must call it
    miou = _miou_from_table(table)

    # keep ONLY the best-by-mIoU checkpoint (full framework ckpt, ~1.3GB, with
    # optimizer + 'model' key so eval.py can load it just like author/Best.pth)
    is_master = (not dist.is_initialized()) or dist.get_rank() == 0
    if miou is not None:
        best = getattr(self, '_best_miou', -1.0)
        if miou > best:
            self._best_miou = miou
            if is_master:
                self.checkpoint.save(filename='Best.pth')
                self.logger.info('[best-ckpt] new best mIoU=%.5f @ step %d -> Best.pth'
                                 % (miou, self.checkpoint.global_step))
        elif is_master:
            self.logger.info('[best-ckpt] mIoU=%.5f (best stays %.5f @ step kept)'
                             % (miou, self._best_miou))

    torch.cuda.empty_cache()


def register_evaluate_fn(launcher):
    launcher.override_evaluate(evaluate_cls_fn)


def seed_torch(seed=2333):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False


if __name__ == '__main__':
    seed_torch(2333)
    trainer = er.trainer.get_trainer()()
    trainer.run(after_construct_launcher_callbacks=[register_evaluate_fn])
