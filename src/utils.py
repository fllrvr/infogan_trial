import os
import numpy as np
import scipy.misc
import matplotlib.pyplot as plt


def plot_loss(iter_list, g_loss_list, d_loss_list,
                 save_dir_path, fname):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(iter_list, g_loss_list, label='Generator')
    ax.plot(iter_list, d_loss_list, label='Discriminator')
    ax.legend()
    ax.set_title('Loss')
    ax.set_xlabel('iterations')
    ax.set_ylabel('loss')
    plt.savefig(os.path.join(save_dir_path, fname))
    plt.close()


def visualize_results(generator, sample_num, sample_vars,
                                  save_dir, save_file_name):
    generator.eval()

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, save_file_name)

    image_frame_dim = int(np.floor(np.sqrt(sample_num)))
    sample_z_, sample_c_, sample_y_ = sample_vars
    
    # vary variables
    samples = generator(sample_z_, sample_c_, sample_y_)
    if samples.is_cuda:
        samples = samples.cpu().data.numpy().transpose(0, 2, 3, 1)
    else:
        samples = samples.data.numpy().transpose(0, 2, 3, 1)

    samples = (samples + 1) / 2
        
    save_images(
        samples[:image_frame_dim * image_frame_dim, :, :, :],
        [image_frame_dim, image_frame_dim],
        save_path)


def save_images(images, size, image_path):
    image = np.squeeze(merge(images, size))
    plt.tick_params(labelbottom=False, labelleft=False,
                              labelright=False,labeltop=False,
                              bottom=False, left=False, right=False,top=False)
    plt.imshow(image, cmap='gray')
    plt.savefig(image_path, bbox_inches='tight')
    plt.close()


def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    if images.shape[3] in (3, 4):
        c = images.shape[3]
        img = np.zeros((h * size[0], w * size[1], c))
        for idx, image in enumerate(images):
            i = idx // size[1]
            j = idx % size[1]
            img[i * h:(i+1) * h, j * w:(j+1) * w, :] = image
        return img
            
    elif images.shape[3] == 1:
        img = np.zeros((h * size[0], w * size[1]))
        for idx, image in enumerate(images):
            i = idx // size[1]
            j = idx % size[1]
            img[i * h:(i+1) * h, j * w:(j+1) * w] = image[:, : ,0]
        return img
    
    else:
        raise ValueError('in merge(images, size) images parameter'
                                   'must have dimensions (h, w), (h, w, 3) or (h, w, 4).')

