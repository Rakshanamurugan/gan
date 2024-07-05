import time
import wandb
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
import matplotlib.pyplot as plt

if __name__ == '__main__':
    opt = TrainOptions().parse()   # get training options
    opt.phase = 'train'
    train_dataset = create_dataset(opt)  # create training dataset
    train_dataset_size = len(train_dataset)  # get the number of images in the training dataset.
    print('The number of training images = %d' % train_dataset_size)

    opt.phase = 'val'
    val_dataset = create_dataset(opt)  # create validation dataset
    val_dataset_size = len(val_dataset)  # get the number of images in the validation dataset.
    print('The number of validation images = %d' % val_dataset_size)

    model = create_model(opt)  # create a model given opt.model and other options
    model.setup(opt)  # regular setup: load and print networks; create schedulers
    visualizer = Visualizer(opt)  # create a visualizer that display/save images and plots
    wandb.init(project='pix2pix', config=opt, name='trail20_trainingimages')

    # # Visualize a few training images to ensure correctness
    # print("Visualizing the first 5 training images...")
    # num_images_to_display = 3
    # for i, data in enumerate(train_dataset):
    #     if i >= num_images_to_display:  # Only visualize the first 5 images
    #         break
    #
    #     # Get image data and filenames
    #     img_A = data['A']
    #     img_B = data['B']
    #     filename = data['A_paths'][0] if opt.direction == 'AtoB' else data['B_paths'][0]
    #
    #     # Convert tensors to numpy arrays and normalize
    #     img_A_np = img_A[0].cpu().numpy().transpose(1, 2, 0)  # Convert from CxHxW to HxWxC
    #     img_B_np = img_B[0].cpu().numpy().transpose(1, 2, 0)  # Convert from CxHxW to HxWxC
    #
    #     # Normalize images to [0, 1] if necessary
    #     img_A_np = (img_A_np - img_A_np.min()) / (img_A_np.max() - img_A_np.min())
    #     img_B_np = (img_B_np - img_B_np.min()) / (img_B_np.max() - img_B_np.min())
    #
    #     # Plot images side by side
    #     plt.figure(figsize=(12, 6))
    #
    #     plt.subplot(1, 2, 1)
    #     plt.imshow(img_A_np, cmap='gray')
    #     plt.title(f'Image A: {filename}')
    #     plt.axis('off')
    #
    #     plt.subplot(1, 2, 2)
    #     plt.imshow(img_B_np, cmap='gray')
    #     plt.title(f'Image B: {filename}')
    #     plt.axis('off')
    #
    #     plt.show()

    total_iters = 0  # the total number of training iterations

    best_val_loss = float('inf')
    best_psnr = -float('inf')
    best_ssim = -float('inf')
    patience = 10
    epochs_no_improve = 0


    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()  # timer for data loading per iteration
        epoch_iter = 0  # the number of training iterations in current epoch, reset to 0 every epoch
        visualizer.reset()  # reset the visualizer: make sure it saves the results to HTML at least once every epoch
        model.update_learning_rate()  # update learning rates in the beginning of every epoch.

        for i, data in enumerate(train_dataset):
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)  # unpack data from dataset and apply preprocessing
            model.optimize_parameters()  # calculate loss functions, get gradients, update network weights

            if total_iters % opt.display_freq == 0:  # display images on visdom and save images to a HTML file
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)
                train_img1 = wandb.Image(model.get_current_visuals()['real_A'])
                train_img2 = wandb.Image(model.get_current_visuals()['real_B'])
                train_img3 = wandb.Image(model.get_current_visuals()['fake_B'])
                wandb.log({"Train Image Real A": train_img1})
                wandb.log({"Train Image Real B": train_img2})
                wandb.log({"Train Image Fake B": train_img3})

            if total_iters % opt.print_freq == 0:  # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                if opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / train_dataset_size, losses)
                wandb.log({"G_GAN_train": losses['G_GAN'],
                           "G_L1_train": losses['G_L1'],
                           "D_real_train": losses['D_real'],
                           "D_fake_train": losses['D_fake']})


            if total_iters % opt.save_latest_freq == 0:  # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()

        # Perform validation at the end of each epoch
        if epoch % opt.val_freq == 0:
            print(f"Performing validation at the end of epoch {epoch}")
            val_losses = model.validate(val_dataset)
            avg_psnr = val_losses['PSNR']
            avg_ssim = val_losses['SSIM']
            wandb.log({"G_GAN_val": val_losses['G_GAN'],
                       "G_L1_val": val_losses['G_L1'],
                       "D_real_val": val_losses['D_real'],
                       "D_fake_val": val_losses['D_fake'],
                       "PSNR_loss": val_losses['PSNR'],
                       "SSIM_loss": val_losses['SSIM']})

            # Check for improvement
            if avg_psnr > best_psnr and avg_ssim > best_ssim:
                best_psnr = avg_psnr
                best_ssim = avg_ssim
                epochs_no_improve = 0
                print(f"New best model saved! PSNR: {avg_psnr:.4f}, SSIM: {avg_ssim:.4f}")
                model.save_networks('best')

                # Log best PSNR and SSIM to wandb
                wandb.log({
                    "best_PSNR": best_psnr,
                    "best_SSIM": best_ssim
                })

            else:
                epochs_no_improve += 1
                if epochs_no_improve == patience:
                    print("Early stopping triggered.")
                    break


        if epoch % opt.save_epoch_freq == 0:  # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))




