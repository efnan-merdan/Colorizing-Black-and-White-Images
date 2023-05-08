import argparse
from methods import *
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser() #argparse makes the command-line interface more friendly and points out when an error occurs
parser.add_argument('-i', '--img_path', type=str, default='imgs/oldgranny.jpeg') #name, flag, type, default(image destination)
parser.add_argument('--use_gpu', action='store_true', help='whether to use GPU') #flag, action, help
parser.add_argument('-o', '--save_prefix', type=str, default='saved', help='will save into this file with colorizer suffixes') #name, flag, type, default, help
option = parser.parse_args() #Convert argument strings to objects and assign them as attributes of the namespace. Return the populated namespace.

colorizer_eccv16 = eccv16(pretrained=True).eval()
colorizer_mefnan23 = mefnan23(pretrained=True).eval()

# What is cude
# torch.cuda is used to set up and run CUDA operations
# It keeps track of the currently selected GPU, and all CUDA tensors you allocate will by default be created on that device

if(option.use_gpu):
    colorizer_eccv16.cuda()
    colorizer_mefnan23.cuda()

img = load_img(option.img_path)
(tens_l_orig, tens_l_rs) = preprocess_img(img, HW=(256,256))

if (option.use_gpu):
    tens_l_rs = tens_l_rs.cuda

img_bw = postprocess_tens(tens_l_orig, torch.cat((0 * tens_l_orig, 0 * tens_l_orig), dim=1))
out_img_eccv16 = postprocess_tens(tens_l_orig, colorizer_eccv16(tens_l_rs).cpu())
out_img_mefnan23 = postprocess_tens(tens_l_orig, colorizer_mefnan23(tens_l_rs).cpu())


plt.imsave('%s_eccv16.jpeg'%option.save_prefix, out_img_eccv16)
plt.imsave('%s_mefnan23.jpeg'%option.save_prefix, out_img_mefnan23)

plt.figure(figsize=(12,8))
plt.subplot(2,2,1)
plt.imshow(img)
plt.title('Original')
plt.axis('off')


plt.subplot(2,2,2)
plt.imshow(img_bw)
plt.title('Input')
plt.axis('off')


plt.subplot(2,2,3)
plt.imshow(out_img_eccv16)
plt.title('ECCV16 Method')
plt.axis('off')

plt.subplot(2,2,4)
plt.imshow(out_img_mefnan23)
plt.title('MEFNAN23 Method')
plt.axis('off')
plt.show()



