# import os
# import numpy as np
# import cv2
# import matplotlib.pyplot as plt
# import nibabel as nib
# from options.test_options import TestOptions
# from data import create_dataset
# from models import create_model
# from util.visualizer import save_images
# from util import html
# from skimage.metrics import peak_signal_noise_ratio
#
# try:
#     import wandb
# except ImportError:
#     print('Warning: wandb package cannot be found. The option "--use_wandb" will result in error.')
#
#
# def calculate_psnr(img1, img2):
#     img1 = img1.astype(np.float32)
#     img2 = img2.astype(np.float32)
#     mse = np.mean((img1 - img2) ** 2)
#     if mse == 0:
#         return float('inf')
#     max_pixel = np.max(img1)
#     psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
#     return psnr
#
#
# def process_patient_psnr(psnr_values):
#     patient_psnr = {}
#     for filename, psnr in psnr_values:
#         patient_id = filename.split('_')[1]  # Assuming the patient ID is the second part of the filename
#         if patient_id not in patient_psnr:
#             patient_psnr[patient_id] = []
#         patient_psnr[patient_id].append(psnr)
#
#     # Calculate average PSNR for each patient
#     average_patient_psnr = {patient: np.mean(values) for patient, values in patient_psnr.items()}
#     return average_patient_psnr
#
#
# def stack_slices_to_3d(slices, affine=None):
#     """
#     Stack 2D slices to create a 3D volume and save as NIfTI.
#     """
#     volume = np.stack(slices, axis=-1)
#     if affine is None:
#         affine = np.eye(4)
#     nifti_img = nib.Nifti1Image(volume, affine)
#     return nifti_img
#
#
# def save_nifti(nifti_img, filepath):
#     """
#     Save a NIfTI image to file.
#     """
#     nib.save(nifti_img, filepath)
#
#
# if __name__ == '__main__':
#     opt = TestOptions().parse()  # get test options
#     opt.phase = 'train'  # Ensure the phase is set to 'train'
#     opt.num_threads = 0  # test code only supports num_threads = 0
#     opt.batch_size = 1  # test code only supports batch_size = 1
#     opt.serial_batches = True  # disable data shuffling
#     opt.no_flip = True  # no flip
#     opt.display_id = -1  # no visdom display
#
#     dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
#     dataset_size = len(dataset)  # get the number of images in the dataset.
#     print('The number of training images = %d' % dataset_size)
#     model = create_model(opt)  # create a model given opt.model and other options
#     model.setup(opt)  # regular setup: load and print networks; create schedulers
#
#     if opt.use_wandb:
#         wandb_run = wandb.init(project=opt.wandb_project_name, name=opt.name,
#                                config=opt) if not wandb.run else wandb.run
#         wandb_run._label(repo='CycleGAN-and-pix2pix')
#
#     web_dir = os.path.join(opt.results_dir, opt.name,
#                            '{0}_{1}'.format(opt.phase, opt.epoch))  # define the website directory
#     if opt.load_iter > 0:
#         web_dir = '{0}_iter{1}'.format(web_dir, opt.load_iter)
#     print('creating web directory', web_dir)
#     webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
#
#     if opt.eval:
#         model.eval()
#
#     psnr_values = []
#     reconstructed_slices = []
#     real_slices = []
#     for i, data in enumerate(dataset):
#         model.set_input(data)  # unpack data from data loader
#         model.test()  # run inference
#         visuals = model.get_current_visuals()  # get image results
#         img_path = model.get_image_paths()  # get image paths
#
#         # Load the original and generated images
#         original_img = data['A'].squeeze().cpu().numpy()
#         generated_img = visuals['fake_B'].squeeze().cpu().numpy()
#
#         # Calculate PSNR for 2D slices
#         psnr = calculate_psnr(original_img, generated_img)
#         psnr_values.append((os.path.basename(img_path[0]), psnr))
#
#         # Store slices for 3D reconstruction
#         reconstructed_slices.append(generated_img)
#         real_slices.append(original_img)
#
#
#
#     webpage.save()
#
#     # Stack slices to create 3D volumes
#     reconstructed_3d = stack_slices_to_3d(reconstructed_slices)
#     real_3d = stack_slices_to_3d(real_slices)
#
#     # Save 3D volumes as NIfTI
#     save_nifti(reconstructed_3d, os.path.join(web_dir, 'reconstructed_3d.nii.gz'))
#     save_nifti(real_3d, os.path.join(web_dir, 'real_3d.nii.gz'))
#
#     # Calculate PSNR for 3D volumes
#     psnr_3d = calculate_psnr(real_3d.get_fdata(), reconstructed_3d.get_fdata())
#     print('3D PSNR: %.4f' % psnr_3d)
#
#     # Process and print the average PSNR for each patient (2D slices)
#     average_patient_psnr = process_patient_psnr(psnr_values)
#     for patient, avg_psnr in average_patient_psnr.items():
#         print('Patient {0}: Average 2D PSNR = {1:.4f}'.format(patient, avg_psnr))


import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import nibabel as nib
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html

try:
    import wandb
except ImportError:
    print('Warning: wandb package cannot be found. The option "--use_wandb" will result in error.')


def calculate_psnr(img1, img2):
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = np.max(img1)
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr


def process_patient_psnr(psnr_values):
    patient_psnr = {}
    for filename, psnr in psnr_values:
        patient_id = filename.split('_')[1]  # Assuming the patient ID is the second part of the filename
        if patient_id not in patient_psnr:
            patient_psnr[patient_id] = []
        patient_psnr[patient_id].append(psnr)

    # Calculate average PSNR for each patient
    average_patient_psnr = {patient: np.mean(values) for patient, values in patient_psnr.items()}
    return average_patient_psnr


def stack_slices_to_3d(slices, affine=None):
    """
    Stack 2D slices to create a 3D volume and save as NIfTI.
    """
    volume = np.stack(slices, axis=-1)
    if affine is None:
        affine = np.eye(4)
    nifti_img = nib.Nifti1Image(volume, affine)
    return nifti_img


def save_nifti(nifti_img, filepath):
    """
    Save a NIfTI image to file.
    """
    nib.save(nifti_img, filepath)


if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    opt.phase = 'train'  # Ensure the phase is set to 'train'
    opt.num_threads = 0  # test code only supports num_threads = 0
    opt.batch_size = 1  # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling
    opt.no_flip = True  # no flip
    opt.display_id = -1  # no visdom display

    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)  # get the number of images in the dataset.
    print('The number of inference images = %d' % dataset_size)
    model = create_model(opt)  # create a model given opt.model and other options
    model.setup(opt)  # regular setup: load and print networks; create schedulers

    if opt.use_wandb:
        wandb_run = wandb.init(project=opt.wandb_project_name, name=opt.name,
                               config=opt) if not wandb.run else wandb.run
        wandb_run._label(repo='CycleGAN-and-pix2pix')

    web_dir = os.path.join(opt.results_dir, opt.name,
                           '{0}_{1}'.format(opt.phase, opt.epoch))  # define the website directory
    if opt.load_iter > 0:
        web_dir = '{0}_iter{1}'.format(web_dir, opt.load_iter)
    print('creating web directory', web_dir)
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))

    if opt.eval:
        model.eval()

    psnr_values = []
    patient_slices = {}
    for i, data in enumerate(dataset):
        model.set_input(data)  # unpack data from data loader

        img_path = model.get_image_paths()  # get image paths
        print(f"Processing file: {os.path.basename(img_path[0])}")

        # Print shapes of input tensors
        print(f"Shape of input tensor 'A': {data['A'].shape}")
        print(f"Shape of input tensor 'B': {data['B'].shape}")

        model.test()  # run inference
        visuals = model.get_current_visuals()  # get image results

        # Load the original and generated images
        original_img = data['A'].squeeze().cpu().numpy()
        generated_img = visuals['fake_B'].squeeze().cpu().numpy()

        # Calculate PSNR for 2D slices
        psnr = calculate_psnr(original_img, generated_img)
        psnr_values.append((os.path.basename(img_path[0]), psnr))

        # Extract patient ID
        patient_id = os.path.basename(img_path[0]).split('_')[
            1]  # Assuming the patient ID is the second part of the filename

        # Store slices by patient ID for 3D reconstruction
        if patient_id not in patient_slices:
            patient_slices[patient_id] = {'reconstructed': [], 'real': []}
        patient_slices[patient_id]['reconstructed'].append(generated_img)
        patient_slices[patient_id]['real'].append(original_img)

        # Print individual 2D PSNR
        print('Slice {0}: 2D PSNR = {1:.4f}'.format(i, psnr))

    webpage.save()

    # Process and print the average PSNR for each patient (2D slices)
    average_patient_psnr = process_patient_psnr(psnr_values)
    for patient, avg_psnr in average_patient_psnr.items():
        print('Patient {0}: Average 2D PSNR = {1:.4f}'.format(patient, avg_psnr))

    # Calculate and print overall average 2D PSNR
    overall_avg_psnr = np.mean([psnr for _, psnr in psnr_values])
    print('Overall Average 2D PSNR: {:.4f}'.format(overall_avg_psnr))

    # Calculate and print 3D PSNR for each patient
    patient_3d_psnr = {}
    for patient_id, slices in patient_slices.items():
        reconstructed_3d = stack_slices_to_3d(slices['reconstructed'])
        real_3d = stack_slices_to_3d(slices['real'])

        # Save 3D volumes as NIfTI
        save_nifti(reconstructed_3d, os.path.join(web_dir, f'reconstructed_3d_patient_{patient_id}.nii.gz'))
        save_nifti(real_3d, os.path.join(web_dir, f'real_3d_patient_{patient_id}.nii.gz'))

        # Calculate PSNR for 3D volumes
        psnr_3d = calculate_psnr(real_3d.get_fdata(), reconstructed_3d.get_fdata())
        patient_3d_psnr[patient_id] = psnr_3d
        print(f'Patient {patient_id}: 3D PSNR = {psnr_3d:.4f}')

    # Save all PSNR results to a file
    with open(os.path.join(web_dir, 'psnr_results.txt'), 'w') as f:
        f.write('Individual 2D PSNR values:\n')
        for i, (filename, psnr) in enumerate(psnr_values):
            f.write('Slice {0} ({1}): {2:.4f}\n'.format(i, filename, psnr))
        f.write('\nPatient-wise Average 2D PSNR:\n')
        for patient, avg_psnr in average_patient_psnr.items():
            f.write('Patient {0}: {1:.4f}\n'.format(patient, avg_psnr))
        f.write('\nOverall Average 2D PSNR: {:.4f}\n'.format(overall_avg_psnr))
        f.write('\nPatient-wise 3D PSNR:\n')
        for patient, psnr_3d in patient_3d_psnr.items():
            f.write('Patient {0}: 3D PSNR = {1:.4f}\n'.format(patient, psnr_3d))

    print('PSNR results saved to:', os.path.join(web_dir, 'psnr_results.txt'))

