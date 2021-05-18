from preprocess import get_dataset, DataLoader, collate_fn_transformer
from network import *
from tensorboardX import SummaryWriter
import torchvision.utils as vutils
import os
from tqdm import tqdm

print('train_transformer.py packages imported..')

# LOADCHECKPOINT = False

def adjust_learning_rate(optimizer, step_num, warmup_step=4000):
    lr = hp.lr * warmup_step**0.5 * min(step_num * warmup_step**-1.5, step_num**-0.5)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

        
def main():
    print('starting here...')
    dataset = get_dataset()
    global_step = 0

    m = nn.DataParallel(Model().cuda())

    # if LOADCHECKPOINT:
    #     m.load_state_dict(t.load(hp.checkpoint_file_transformer))
    #     print('loaded checkpoint...')
    #     m.eval()

    m.train()
    optimizer = t.optim.Adam(m.parameters(), lr=hp.lr)

    pos_weight = t.FloatTensor([5.]).cuda()
    writer = SummaryWriter()
    
    for epoch in range(hp.epochs):
        print('at epoch', epoch)
        dataloader = DataLoader(dataset, batch_size=hp.batch_size, shuffle=True, collate_fn=collate_fn_transformer, drop_last=True, num_workers=1)
        pbar = tqdm(dataloader)
        for i, data in enumerate(pbar):
            pbar.set_description("Processing at epoch %d"%epoch)
            global_step += 1
            if global_step < 400000:
                adjust_learning_rate(optimizer, global_step)
                
            eeg_array, mel, mel_input, pos_eeg_signal, pos_mel, _ = data
            
            stop_tokens = t.abs(pos_mel.ne(0).type(t.float) - 1)
            
            eeg_array = eeg_array.cuda()
            mel = mel.cuda()
            mel_input = mel_input.cuda()
            pos_eeg_signal = pos_eeg_signal.cuda()
            pos_mel = pos_mel.cuda()
            
            print('before m.forward()...')

            mel_pred, postnet_pred, attn_probs, stop_preds, attns_enc, attns_dec = m.forward(eeg_array, mel_input, pos_eeg_signal, pos_mel)

            mel_loss = nn.L1Loss()(mel_pred, mel)
            post_mel_loss = nn.L1Loss()(postnet_pred, mel)
            
            loss = mel_loss + post_mel_loss
            
            writer.add_scalars('training_loss',{
                    'mel_loss':mel_loss,
                    'post_mel_loss':post_mel_loss,

                }, global_step)
                
            writer.add_scalars('alphas',{
                    'encoder_alpha':m.module.encoder.alpha.data,
                    'decoder_alpha':m.module.decoder.alpha.data,
                }, global_step)
            
            
            if global_step % hp.image_step == 1:

                # summarywriter add_image params
                num_images_per_loop = 4
                writer_start_val = int(hp.batch_size / 2)
                writer_end_val = int(hp.batch_size * num_images_per_loop)
                writer_step_val = int(hp.batch_size)
            
                for i, prob in enumerate(attn_probs):
                    num_h = prob.size(0)
                    for j in range(writer_start_val, writer_end_val, writer_step_val):                
                        x = vutils.make_grid([prob[j] * 255])
                        # x  = prob[j] * 255
                        writer.add_image('Attention_%d_0'%global_step, x, i*num_images_per_loop+j)

                for i, prob in enumerate(attns_enc):
                    num_h = prob.size(0)
                    for j in range(writer_start_val, writer_end_val, writer_step_val):
                        x = vutils.make_grid([prob[j] * 255])
                        # x  = prob[j] * 255
                        writer.add_image('Attention_enc_%d_0'%global_step, x, i*num_images_per_loop+j)
            
                for i, prob in enumerate(attns_dec):
                    num_h = prob.size(0)
                    for j in range(writer_start_val, writer_end_val, writer_step_val):
                        x = vutils.make_grid([prob[j] * 255])
                        # x  = prob[j] * 255
                        writer.add_image('Attention_dec_%d_0'%global_step, x, i*num_images_per_loop+j)
                
            optimizer.zero_grad()
            # Calculate gradients
            loss.backward()
            
            nn.utils.clip_grad_norm_(m.parameters(), 1.)
            
            # Update weights
            optimizer.step()

            if global_step % hp.save_step == 0:
                t.save({'model':m.state_dict(),
                        'optimizer':optimizer.state_dict()},
                        os.path.join(hp.checkpoint_path,'checkpoint_transformer_%d.pth.tar' % global_step))

            
            


if __name__ == '__main__':
    # device = t.device("cpu")
    # print('device:')
    # print(device)
    main()