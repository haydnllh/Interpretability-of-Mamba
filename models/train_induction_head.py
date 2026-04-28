import fire
import torch as t
from torch.optim import Adam
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from models.dataloaders.induction_heads import InductionData
from models.configs.induction_heads.config import MambaConfig
from omegaconf import OmegaConf
import logging
import csv
import os

"""
Recreate Experiment in Table 2, Section 4.1.2 and Appendix E1 
"""

def train_induction_head(n_epoch=1000, epoch_sz=8192,
         report_every=100, out_file='model.ckpt', log_dir='/scratch/lhl1g23/lightning_logs/induction_head/'):
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    logger = logging.getLogger()

    hps = OmegaConf.load("models/configs/induction_heads/induction_heads.yaml")
    hps.train.n_epoch = n_epoch
    hps.train.epoch_sz = epoch_sz
    hps.train.report_every = report_every

    data = InductionData(hps.data.batch, hps.data.n_vocab, hps.data.train_len, hps.data.prefix_len)
    mambaconfig = MambaConfig()
    model = MambaLMHeadModel(mambaconfig, device='cuda')

    val_batch = next(iter(InductionData(hps.data.batch, hps.data.n_vocab, hps.data.train_len, hps.data.prefix_len)))
    val_tokens = val_batch['tokens'].to('cuda')

    it = iter(data)
    xent = t.nn.CrossEntropyLoss()
    opt = Adam(model.parameters(), lr=hps.train.learn_rate)
    
    os.makedirs(log_dir, exist_ok=True)
    csv_file = open(log_dir + 'training_log.csv', 'w', newline='')
    writer = csv.writer(csv_file)
    writer.writerow(['epoch', 'step', 'train_loss', 'val_acc'])

    step = 0
    for epoch in range(n_epoch):
        loss_sum = 0
        for b in range(epoch_sz):
            batch = next(it)
            tokens = batch['tokens'].to('cuda')
            opt.zero_grad()
            out = model(tokens[:,:-1]).logits
            pred = out[:,-1,:]
            targ = tokens[:,-1]
            # print(pred.shape, targ.shape)
            # return
            # print(out.shape)
            loss = xent(pred, targ)
            loss.backward()
            loss_sum += loss
            opt.step()

            step += 1
            if step % report_every == 0:
                model.eval()
                with t.no_grad():
                    val_out = model(val_tokens[:,:-1]).logits[:,-1,:]
                    val_targ = val_tokens[:,-1]
                    val_acc = (val_out.argmax(-1) == val_targ).float().mean().item()
                model.train()

                writer.writerow([epoch, step, loss.item(), val_acc])
                csv_file.flush()
                
    csv_file.close() 

    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    state = model.state_dict()
    logger.info(f'Saving to {out_file}')
    t.save(state, out_file)

    # finished training

if __name__ == '__main__':
    fire.Fire(train_induction_head(out_file="/scratch/lhl1g23/checkpoints/induction_heads_new"))
