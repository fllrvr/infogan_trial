import argparse
import os
import shutil
import yaml
import pickle
import time
import itertools
import collections
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from dataset import dataloader_
from network import Generator, Discriminator
from utils import plot_loss, visualize_results
import matplotlib.pyplot as plt


def log_gaussian(x, mu, diag_stddev):
    """
    Return the negative log-likelihood of a Gaussian distribution with diagonal variace.
    """
    logli = - 0.5 * (diag_stddev.mul(2 * np.pi) +1e-10).log() -\
            (x - mu).pow(2).div(diag_stddev.mul(2.0) + 1e-10)
    return logli.sum(1).mean().mul(-1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path')
    parser.add_argument('save_dir_name')
    args = parser.parse_args()
    
    # load configuration file
    with open(args.config_path, 'r') as f:
        config = yaml.load(f)
    if not os.path.exists(args.save_dir_name):
        os.makedirs(args.save_dir_name)
    image_save_dir = os.path.join(args.save_dir_name, 'images')
    if not os.path.exists(image_save_dir):
        os.makedirs(image_save_dir)
    shutil.copy(
        args.config_path, 
        os.path.join(args.save_dir_name, os.path.basename(args.config_path)))
        
    # set dataloader
    dataloader = dataloader_(input_size=config['input_size'],
                             batch_size=config['batch_size'],
                             dataset_name=config['dataset_name'],
                             data_root_dir=config['data_root_dir'])
    data = dataloader.__iter__().__next__()[0]
    
    # initialize generator and discriminator
    G = Generator(input_dim=config['z_dim'],
                  output_dim=data.shape[1],
                  input_size=config['input_size'],
                  len_discrete_code=config['len_discrete_code'],
                  len_continuous_code=config['len_continuous_code'])
    
    D = Discriminator(input_dim=data.shape[1],
                      output_dim=1,
                      input_size=config['input_size'],
                      len_discrete_code=config['len_discrete_code'],
                      len_continuous_code=config['len_continuous_code'])
    
    G_optimizer = optim.Adam(G.parameters(),
                             lr=config['lrG'],
                             betas=(config['beta1'], config['beta2']),
                             amsgrad=True)
    D_optimizer = optim.Adam(D.parameters(),
                             lr=config['lrD'],
                             betas=(config['beta1'], config['beta2']),
                             amsgrad=True)
    info_optimizer = optim.Adam(itertools.chain(G.parameters(), D.parameters()),
                                lr=config['lrD'],
                                betas=(config['beta1'], config['beta2']),
                                amsgrad=True)
    if torch.cuda.is_available():
        G.cuda()
        D.cuda()

    # instances for loss
    BCE_loss = nn.BCELoss()
    CE_loss = nn.CrossEntropyLoss()
    if torch.cuda.is_available():
        BCE_loss.cuda()
        CE_loss.cuda()
    
    # initialize labels
    y_real_ = torch.ones(config['batch_size'], 1)  # not 1 as in the reffered implementation?
    y_fake_ = torch.zeros(config['batch_size'], 1)
    
    if torch.cuda.is_available():
        y_real_ = y_real_.cuda()
        y_fake_ = y_fake_.cuda()
    
    # prepare random sampling
    # for discrete codes
    sample_num_disc = config['sample_per_code'] * config['len_discrete_code']
    sample_z_ = torch.zeros(
        (sample_num_disc, config['z_dim']))
    for i in range(config['sample_per_code']):
        sample_z_[i * config['len_discrete_code']:(i + 1) * config['len_discrete_code']] =\
            torch.rand(1, config['z_dim']).repeat(config['len_discrete_code'], 1)
    
    temp_y = torch.arange(config['len_discrete_code']).view((-1,1)).repeat(config['sample_per_code'], 1)
    sample_y_ = torch.zeros(
        (sample_num_disc, config['len_discrete_code'])).scatter_(1, temp_y.type(torch.LongTensor), 1)
    del temp_y
    sample_c_ = torch.zeros((sample_num_disc, config['len_continuous_code']))
    
    if torch.cuda.is_available():
        sample_z_ = sample_z_.cuda()
        sample_c_ = sample_c_.cuda()
        sample_y_ = sample_y_.cuda()
    
    # for continuous codes
    # assume config['len_continuous_code'] == 2
    sample_num_conti = config['sample_per_code'] ** config['len_continuous_code']
    sample_z2_ = torch.rand((1, config['z_dim'])).expand(sample_num_conti, config['z_dim'])

    temp_c = torch.linspace(-2, 2, config['sample_per_code'])
    sample_c2_ = torch.zeros((sample_num_conti, config['len_continuous_code']))
    sample_c2_[:, 0] = temp_c.repeat(config['sample_per_code'], 1).transpose(1, 0).reshape(-1)
    sample_c2_[:, 1] = temp_c.repeat(config['sample_per_code'], 1).reshape(-1)
    del temp_c

    if torch.cuda.is_available():
        sample_z2_ = sample_z2_.cuda()
        sample_c2_ = sample_c2_.cuda()
    
    # start training
    start_time = time.time()
    D.train()
    train_hist = collections.defaultdict(list)
    iter_num = 0
    for epoch in range(config['epoch']):
        G.train()
        epoch_start_time = time.time()
        for i, (x_, y_) in enumerate(dataloader):
            if i == dataloader.dataset.__len__() // config['batch_size']:
                break
            z_ = torch.rand(config['batch_size'], config['z_dim'])
            if config['supervised']:
                # y_が1の要素を1で置換
                y_disc_ = torch.zeros(
                    (config['batch_size'], config['len_discrete_code'])
                    ).scatter_(1, y_.type(torch.LongTensor).unsqueeze(1), 1)
            else:
                # 0 or 1をmultinomialにしたがってrandomに生成
                y_disc_ = torch.from_numpy(
                    np.random.multinomial(
                        1,
                        config['len_discrete_code'] * [float(1.0 / config['len_discrete_code'])],
                        size=[config['batch_size']])).type(torch.FloatTensor)
                
            y_cont_ = torch.from_numpy(
                np.random.uniform(-1, 1,
                                  size=(config['batch_size'], config['len_continuous_code']))
                                       ).type(torch.FloatTensor)
            if torch.cuda.is_available():
                x_ = x_.cuda()
                z_ = z_.cuda()
                y_disc_ = y_disc_.cuda()
                y_cont_ = y_cont_.cuda()
            
            # update Discriminator
            D_optimizer.zero_grad()
            
            # discriminate real sample
            D_real, _, _, _ = D(x_)
            D_real_loss = BCE_loss(D_real, y_real_)
            
            # discriminate fake sample
            g_ = G(z_, y_cont_, y_disc_)
            D_fake, _, _, _ = D(g_)
            D_fake_loss = BCE_loss(D_fake, y_fake_)
            D_loss = D_real_loss + D_fake_loss

            D_loss.backward(retain_graph=True)
            D_optimizer.step()
            
            # update Generator
            G_optimizer.zero_grad()
            
            g_ = G(z_, y_cont_, y_disc_)
            D_fake, D_disc, D_cont_mean, D_cont_stddev = D(g_)
            
            # The following loss is equal to - log(D(G(z)))
            # while the original loss is equal to log(1- D(G(z))).
            #TODO: check the sizes
            G_loss = BCE_loss(D_fake, y_real_)
            
            # Add a variantional lower bound minus H(c) (=const)
            disc_loss = CE_loss(D_disc, torch.max(y_disc_, 1)[1])   # input: D_discと, index
            cont_loss = log_gaussian(y_cont_, D_cont_mean, D_cont_stddev)
            if torch.cuda.is_available():
                cont_loss = cont_loss.cuda()
            info_loss = config['lambda'] * (disc_loss + cont_loss)
            G_loss += info_loss
    
            G_loss.backward(retain_graph=True)
            G_optimizer.step()
            
            iter_num += 1
            
            # report
            if iter_num % config['report_interval_iter'] == 0:
                train_hist['D_loss'].append(D_loss.item())
                train_hist['info_loss'].append(info_loss.item())
                train_hist['G_loss'].append(G_loss.item())
                train_hist['iteration'].append(iter_num + 1)
                train_hist['epoch'].append(epoch + 1)
                train_hist['elapsed_time'].append(time.time() - start_time)
                print(
                '''Epoch: {}/{},  Iteration: {}, D_loss: {:.4f}, G_loss: {:.4f}, info_loss: {:.4f}
                '''.format((epoch + 1), (config['epoch']), iter_num,
                              D_loss.item(), G_loss.item(), info_loss.item()))
                
                # save train log
                with open(os.path.join(args.save_dir_name, 'train_history.pkl'), 'wb') as f:
                    pickle.dump(train_hist, f)
                
                # save loss plot
                plot_loss(iter_list=train_hist['iteration'],
                             g_loss_list=train_hist['G_loss'],
                             d_loss_list=train_hist['D_loss'],
                            save_dir_path=args.save_dir_name,
                            fname='loss.png')
                
            # save sample figure
            if iter_num % config['save_fig_interval_iter'] == 0:
                # vary discrete codes
                sample_vars_disc = (sample_z_, sample_c_, sample_y_)
                fname = 'sample_disc_{}.png'.format(iter_num)
                visualize_results(G,
                                            sample_num=sample_num_disc,
                                            sample_vars=sample_vars_disc,
                                            save_dir=image_save_dir,
                                            save_file_name=fname)
                
                # vary continuous codes
                for disc_code_idx in range(config['len_discrete_code']):
                    sample_y2_ = torch.zeros(sample_num_conti, config['len_discrete_code'])
                    sample_y2_[:, disc_code_idx] = 1

                    if torch.cuda.is_available():
                        sample_y2_ = sample_y2_.cuda()

                    sample_vars_conti = (sample_z2_, sample_c2_, sample_y2_)
                    fname = 'sample_contis_disc_{}_{}.png'.format(disc_code_idx, iter_num)
                    visualize_results(G,
                                                sample_num=sample_num_conti,
                                                sample_vars=sample_vars_conti,
                                                save_dir=image_save_dir,
                                                save_file_name=fname)
                
        # save model
        if (epoch + 1) % config['snapshot_interval_epoch'] == 0:
            torch.save(G.state_dict(),
                       os.path.join(args.save_dir_name,
                       'generator_{}'.format(epoch + 1)))
#             torch.save(D.state_dict(),
#                        os.path.join(args.save_dir_name,
#                        'discriminator_{}'.format(epoch + 1)))

    # save the last figure
    # vary discrete codes
    sample_vars_disc = (sample_z_, sample_c_, sample_y_)
    fname = 'sample_disc_{}.png'.format(iter_num)
    visualize_results(G,
                                sample_num=sample_num_disc,
                                sample_vars=sample_vars_disc,
                                save_dir=image_save_dir,
                                save_file_name=fname)

    # vary continuous codes
    for disc_code_idx in range(config['len_discrete_code']):
        sample_y2_ = torch.zeros(sample_num_conti, config['len_discrete_code'])
        sample_y2_[:, disc_code_idx] = 1

        if torch.cuda.is_available():
            sample_y2_ = sample_y2_.cuda()

        sample_vars_conti = (sample_z2_, sample_c2_, sample_y2_)
        fname = 'sample_contis_disc_{}_{}.png'.format(disc_code_idx, iter_num)
        visualize_results(G,
                                    sample_num=sample_num_conti,
                                    sample_vars=sample_vars_conti,
                                    save_dir=image_save_dir,
                                    save_file_name=fname)

if __name__ == "__main__":
    main()
