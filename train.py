import lightning as L
from lightning.pytorch.callbacks import ThroughputMonitor
from lightning.fabric.utilities.throughput import measure_flops
from MHBAMixerV2.utils import Text8DataModule, MHBAMixerV2Module
import torch
from MHBAMixerV2.config_MHBAMixerV2 import MHBAMixerV2Config
from lightning.pytorch.callbacks import ModelCheckpoint

torch.set_float32_matmul_precision('medium')
torch.manual_seed(0)


MHBAMixerV2config = MHBAMixerV2Config()

checkpoint_callback = ModelCheckpoint(monitor="val_loss", 
                                    filename='mixer-best-{epoch:03d}-{val_acc:.3f}',
                                    save_top_k=1,
                                    mode='max',
                                    save_last=True
                                    )
trainer = L.Trainer(**MHBAMixerV2config.trainer, callbacks=[checkpoint_callback])
dm = Text8DataModule(64)
model = MHBAMixerV2Module(MHBAMixerV2config)
trainer.fit(model, datamodule=dm)