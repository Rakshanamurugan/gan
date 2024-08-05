# """General-purpose test script for image-to-image translation.
#
# Once you have trained your model with train.py, you can use this script to test the model.
# It will load a saved model from '--checkpoints_dir' and save the results to '--results_dir'.
#
# It first creates model and dataset given the option. It will hard-code some parameters.
# It then runs inference for '--num_test' images and save results to an HTML file.
#
# Example (You need to train models first or download pre-trained models from our website):
#     Test a CycleGAN model (both sides):
#         python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
#
#     Test a CycleGAN model (one side only):
#         python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout
#
#     The option '--model test' is used for generating CycleGAN results only for one side.
#     This option will automatically set '--dataset_mode single', which only loads the images from one set.
#     On the contrary, using '--model cycle_gan' requires loading and generating results in both directions,
#     which is sometimes unnecessary. The results will be saved at ./results/.
#     Use '--results_dir <directory_path_to_save_result>' to specify the results directory.
#
#     Test a pix2pix model:
#         python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA
#
# See options/base_options.py and options/test_options.py for more test options.
# See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
# See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
# """
# import os
# from options.test_options import TestOptions
# from data import create_dataset
# from models import create_model
# from util.visualizer import save_images
# from util import html
#
# try:
#     import wandb
# except ImportError:
#     print('Warning: wandb package cannot be found. The option "--use_wandb" will result in error.')
#
#
# if __name__ == '__main__':
#     opt = TestOptions().parse()  # get test options
#     opt.phase='test'
#     # hard-code some parameters for test
#     opt.num_threads = 0   # test code only supports num_threads = 0
#     opt.batch_size = 1    # test code only supports batch_size = 1
#     opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
#     opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
#     opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
#     dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
#     dataset_size = len(dataset)  # get the number of images in the dataset.
#     print('The number of testing images = %d' % dataset_size)
#     model = create_model(opt)      # create a model given opt.model and other options
#     model.setup(opt)               # regular setup: load and print networks; create schedulers
#
#     # initialize logger
#     if opt.use_wandb:
#         wandb_run = wandb.init(project=opt.wandb_project_name, name=opt.name, config=opt) if not wandb.run else wandb.run
#         wandb_run._label(repo='CycleGAN-and-pix2pix')
#
#     # create a website
#     web_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.epoch))  # define the website directory
#     if opt.load_iter > 0:  # load_iter is 0 by default
#         web_dir = '{:s}_iter{:d}'.format(web_dir, opt.load_iter)
#     print('creating web directory', web_dir)
#     webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
#     # test with eval mode. This only affects layers like batchnorm and dropout.
#     # For [pix2pix]: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
#     # For [CycleGAN]: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.
#     if opt.eval:
#         model.eval()
#     for i, data in enumerate(dataset):
#         if i >= opt.num_test:  # only apply our model to opt.num_test images.
#             break
#         model.set_input(data)  # unpack data from data loader
#         model.test()           # run inference
#         visuals = model.get_current_visuals()  # get image results
#         img_path = model.get_image_paths()     # get image paths
#         if i % 5 == 0:  # save images to an HTML file
#             print('processing (%04d)-th image... %s' % (i, img_path))
#         save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize, use_wandb=opt.use_wandb)
#     webpage.save()  # save the HTML

import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html
from skimage.metrics import peak_signal_noise_ratio as psnr
import numpy as np
from collections import defaultdict

try:
    import wandb
except ImportError:
    print('Warning: wandb package cannot be found. The option "--use_wandb" will result in error.')

def calculate_psnr(reconstructed, real):
    """Calculate PSNR between two images."""
    # Convert images to the range [0, 255] if they are in the range [0, 1]
    if reconstructed.max() <= 1.0:
        reconstructed = (reconstructed * 255).astype(np.uint8)
    if real.max() <= 1.0:
        real = (real * 255).astype(np.uint8)
    return psnr(reconstructed, real)

if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    opt.phase = 'train'  # Ensure the phase is set to 'test'

    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)  # get the number of images in the dataset.
    print('The number of testing images = %d' % dataset_size)

    #set to test mode
    opt.phase = 'test'

    # hard-code some parameters for test
    opt.num_threads = 0  # test code only supports num_threads = 0
    opt.batch_size = 1  # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True  # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1  # no visdom display; the test code saves the results to a HTML file.

    # dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    # dataset_size = len(dataset)  # get the number of images in the dataset.
    # print('The number of testing images = %d' % dataset_size)

    model = create_model(opt)  # create a model given opt.model and other options
    model.setup(opt)  # regular setup: load and print networks; create schedulers

    # initialize logger
    if opt.use_wandb:
        wandb_run = wandb.init(project=opt.wandb_project_name, name=opt.name,
                               config=opt) if not wandb.run else wandb.run
        wandb_run._label(repo='CycleGAN-and-pix2pix')

    # create a website
    web_dir = os.path.join(opt.results_dir, opt.name,
                           '{}_{}'.format(opt.phase, opt.epoch))  # define the website directory
    if opt.load_iter > 0:  # load_iter is 0 by default
        web_dir = '{:s}_iter{:d}'.format(web_dir, opt.load_iter)
    print('creating web directory', web_dir)
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))

    # test with eval mode. This only affects layers like batchnorm and dropout.
    if opt.eval:
        model.eval()

    psnr_values = []  # List to store PSNR values
    patient_psnr = defaultdict(list)  # Dictionary to store PSNR values for each patient

    for i, data in enumerate(dataset):
        model.set_input(data)  # unpack data from data loader
        model.test()  # run inference
        visuals = model.get_current_visuals()  # get image results
        img_path = model.get_image_paths()  # get image paths

        # Extract reconstructed and real images
        reconstructed_img = visuals['fake_B'][0, 0].cpu().numpy()  # Assuming the key for reconstructed image is 'fake_B'
        real_img = visuals['real_B'][0, 0].cpu().numpy()  # Assuming the key for real image is 'real_B'

        # Calculate PSNR
        psnr_value = calculate_psnr(reconstructed_img, real_img)
        psnr_values.append((img_path[0], psnr_value))

        # Extract patient ID from image path (assuming the patient ID is part of the filename)
        patient_id = os.path.basename(img_path[0]).split('_')[1]
        patient_psnr[patient_id].append(psnr_value)

        # Print PSNR and filename
        print(f'Image {i} ({img_path[0]}): PSNR = {psnr_value}')

        save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize,
                    use_wandb=opt.use_wandb)

    # Calculate average PSNR for each patient
    patient_avg_psnr = {patient: np.mean(values) for patient, values in patient_psnr.items()}

    # Find the patient with the best and worst average PSNR
    best_patient = max(patient_avg_psnr, key=patient_avg_psnr.get)
    worst_patient = min(patient_avg_psnr, key=patient_avg_psnr.get)

    # Print average PSNR for each patient
    for patient, avg_psnr in patient_avg_psnr.items():
        print(f'Patient {patient}: Average PSNR = {avg_psnr}')

    # Print the patient with the best and worst average PSNR
    print(f'Best Patient: {best_patient} with Average PSNR = {patient_avg_psnr[best_patient]}')
    print(f'Worst Patient: {worst_patient} with Average PSNR = {patient_avg_psnr[worst_patient]}')

    # Save PSNR values to a file
    with open(os.path.join(web_dir, 'psnr_values.txt'), 'w') as f:
        for img_path, value in psnr_values:
            f.write(f'Image ({img_path}): PSNR = {value}\n')
        f.write('\nAverage PSNR for each patient:\n')
        for patient, avg_psnr in patient_avg_psnr.items():
            f.write(f'Patient {patient}: Average PSNR = {avg_psnr}\n')
        f.write(f'\nBest Patient: {best_patient} with Average PSNR = {patient_avg_psnr[best_patient]}\n')
        f.write(f'Worst Patient: {worst_patient} with Average PSNR = {patient_avg_psnr[worst_patient]}\n')

    webpage.save()  # save the HTML






