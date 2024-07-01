from PIL import Image

# Open the image file
img = Image.open('/media/e210-pc34/System/rakshana/pytorch-CycleGAN-and-pix2pix/results/mriimagetranslation2/test_latest/images/BraTS2021_00009_slice_021_real_A.png')

# Print the dimensions of the image
print(img.size)