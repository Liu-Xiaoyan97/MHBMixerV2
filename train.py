import lightning as L
from lightning.pytorch.callbacks import ThroughputMonitor
from lightning.fabric.utilities.throughput import measure_flops
from MHBAMixerV2.utils import MHBAMixerV2DataModule, MHBAMixerV2Module
import torch
from MHBAMixerV2.config_MHBAMixerV2 import MHBAMixerV2Config
from lightning.pytorch.callbacks import ModelCheckpoint
import argparse
torch.set_float32_matmul_precision('medium')
torch.manual_seed(0)
import os 

os.environ["TOKENIZERS_PARALLELISM"] = "false"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="funeturing from checkpoint")
    parser.add_argument("--ckpt", type=str, default=None, help="model ckpt path for funeturing")
    parser.add_argument("--onlytest", action="store_true", help="If you only want to perform a test, please enable this field.")
    MHBAMixerV2config = MHBAMixerV2Config()

    checkpoint_callback = ModelCheckpoint(monitor="val_loss", 
                                        filename='mixer-best-{epoch:03d}-{val_loss:.3f}',
                                        save_top_k=1,
                                        mode='min',
                                        save_last=True
                                        )
    args = parser.parse_args()
    trainer = L.Trainer(**MHBAMixerV2config.trainer, callbacks=[checkpoint_callback])
    dm = MHBAMixerV2DataModule(MHBAMixerV2config.batch_size)
    model = MHBAMixerV2Module(MHBAMixerV2config)
    if args.ckpt is not None:
        model = MHBAMixerV2Module.load_from_checkpoint(args.ckpt)
    if args.onlytest:
        dm.setup("test")
        trainer.test(model, datamodule=dm)
    else:
        dm.setup("fit")
        trainer.fit(model, datamodule=dm)
        

