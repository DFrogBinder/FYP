import logging
import os
import shutil
import time
import matplotlib.pyplot as plt
import numpy as np

import SimpleITK as sitk
import torch
import torch.nn.functional as F
import torchio as tio
import torchvision
from tensorboardX import SummaryWriter

import model.loss
import model.regnet
import model.util
from agents.base import BaseAgent
from model.util import count_parameters
from utils.SpatialTransformer import SpatialTransformer
from utils.data_loader import data_loader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class groupAgent(BaseAgent):
    def __init__(self, args):
        super(groupAgent).__init__()

        self.args = args
        self.logger = logging.getLogger()
        # initialize my counters
        self.current_epoch = 0

        if self.args.mode == 'eval':
            pass
        else:

            # initialize tensorboard writer
            self.summary_writer = SummaryWriter(self.args.tensorboard_dir)

            # Create an instance from the data loader
            if self.args.debug:
                _, _, _, self.train_loader = data_loader(debug_data_folder=self.args.validation_data_folder,
                                                    num_workers=self.args.num_workers, aug_type=self.args.aug_type)
                self.validation_loader = self.train_loader
                self.test_loader = self.train_loader
            else:
                self.train_loader, self.validation_loader, self.test_loader, _ = data_loader(
                    train_data_folder=self.args.train_data_folder,
                    validation_data_folder=self.args.validation_data_folder,
                    test_data_folder=self.args.test_data_folder, num_workers=self.args.num_workers,
                    aug_type=self.args.aug_type)

            # Create an instance from the Model
            self.model = model.regnet.RegNet_single(dim=self.args.dim, n=self.args.num_images_per_group,
                                                    scale=self.args.scale, depth=self.args.depth,
                                                    initial_channels=self.args.initial_channels,
                                                    normalization=self.args.normalization).to(device)

            self.logger.info(self.model)
            self.logger.info(f"Total Trainable Params: {count_parameters(self.model)}")

            # Create instance from the loss
            self.ncc_loss = model.loss.LNCC(self.args.dim, self.args.ncc_window_size).to(device)
            self.spatial_transform = SpatialTransformer(dim=self.args.dim)

            # Create instance from the optimizer
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)

            # Model Loading from the latest checkpoint if not found start from scratch.
            self.load_checkpoint()

    def save_checkpoint(self, filename='model.pth.tar', is_best=False):
        """
        Saving the latest checkpoint of the training
        :param filename: filename which will contain the state
        :param is_best: flag is it is the best model
        :return:
        """
        state = {
            'epoch': self.current_epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict()}
        # Save the state
        torch.save(state, os.path.join(self.args.model_dir, filename))
        # If it is the best copy it to another file 'model_best.pth.tar'
        if is_best:
            shutil.copyfile(os.path.join(self.args.model_dir, filename),
                            os.path.join(self.args.model_dir, 'model_best.pth.tar'))

    def load_checkpoint(self, filename='model.pth.tar'):
        filename = os.path.join(self.args.model_dir, filename)
        try:
            self.logger.info("Loading checkpoint '{}'".format(filename))
            checkpoint = torch.load(filename, map_location=device)

            self.current_epoch = checkpoint['epoch']
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])

            self.logger.info("Model loaded successfully from '{}' at (epoch {}) \n"
                             .format(self.args.model_dir, checkpoint['epoch']))
        except OSError as e:
            self.logger.info("No model exists from '{}'. Skipping...".format(self.args.model_dir))
            self.logger.info("**First time to train**")

    def run(self):
        """
        This function will the operator
        :return:
        """
        try:
            if self.args.mode == 'train':
                self.train()
            elif self.args.mode == 'inference':
                self.inference()
            elif self.args.mode == 'eval':
                self.eval()

        except KeyboardInterrupt:
            self.logger.info("You have entered CTRL+C.. Wait to finalize")

    def train(self):
        since = time.time()

        for epoch in range(self.current_epoch, self.args.num_epochs):

            self.logger.info('-' * 10)
            self.logger.info('Epoch {}/{}'.format(epoch, self.args.num_epochs))

            self.current_epoch = epoch
            self.train_one_epoch()

            if (epoch) % self.args.validate_every == 0:
                self.validate()

            self.save_checkpoint()

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        self.logger.info('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    def train_one_epoch(self):

        # Set model to training mode
        self.model.train()
        # initialize stats
        running_total_loss = 0.
        running_simi_loss = 0.
        running_cyclic_loss = 0.
        running_smooth_loss = 0.
        running_ncc_per_frame = [0. for x in range(self.args.num_images_per_group)]

        for train_batch in self.train_loader:
            # switch model to training mode, clear gradient accumulators
            self.model.train()
            self.optimizer.zero_grad()
            total_loss = 0.

            batch_init = train_batch['image'][tio.DATA].to(device)[0, ...]
            batch_resampled = F.interpolate(batch_init.unsqueeze(0).type(torch.float32), (self.args.image_shape[0],
                                                                              self.args.image_shape[1],
                                                                              self.args.num_images_per_group),
                                                mode='trilinear', align_corners=True).squeeze(0)

            batch_mri = batch_resampled.permute(3, 0, 2, 1)  # (n,1,h,w)

            # # to visualize the grid
            # grid = torchvision.utils.make_grid(batch_mri, nrow=5)
            # plt.imshow(grid.cpu().permute(1, 2, 0)); plt.axis('off')
            # plt.imsave("Image.png", np.asarray(grid.cpu().permute(1, 2, 0)), format='png')
            # print(batch_resampled.shape)

            # Forward pass
            res = self.model(batch_mri)

            # Similarity loss
            simi_loss, ncc_per_frame = self.ncc_loss(res['warped_input_image'], res['template'], device=device)
            simi_loss_item = simi_loss.item()
            total_loss += simi_loss

            # Smoothness loss
            if self.args.smooth_reg > 0:
                if self.args.smooth_reg_type == 'dvf':
                    smooth_loss = model.loss.smooth_loss_simple(res['scaled_disp_t2i'], res['scaled_template'])
                else:
                    smooth_loss = model.loss.smooth_loss(res['scaled_disp_t2i'], res['scaled_template'])
                total_loss += self.args.smooth_reg * smooth_loss
                smooth_loss_item = smooth_loss.item()
            else:
                smooth_loss_item = 0

            # Cyclic loss
            if self.args.cyclic_reg > 0:
                cyclic_loss = (torch.mean((torch.sum(res['scaled_disp_t2i'], 0)) ** 2)) ** 0.5
                total_loss += self.args.cyclic_reg * cyclic_loss
                cyclic_loss_item = cyclic_loss.item()
            else:
                cyclic_loss_item = 0

            # Collect losses for tensorboard
            running_ncc_per_frame = [x + y for x, y in zip(running_ncc_per_frame, list(ncc_per_frame.cpu().detach().numpy()))]
            running_simi_loss += simi_loss_item
            running_smooth_loss += smooth_loss_item
            running_cyclic_loss += cyclic_loss_item

            total_loss_item = total_loss.item()
            running_total_loss += total_loss_item

            # Backpropagate and update optimizer learning rate
            total_loss.backward()
            self.optimizer.step()

        epoch_total_loss = running_total_loss / len(self.train_loader)
        epoch_simi_loss = running_simi_loss / len(self.train_loader)
        epoch_smooth_loss = running_smooth_loss / len(self.train_loader)
        epoch_cyclic_loss = running_cyclic_loss / len(self.train_loader)
        epoch_ncc_per_frame = [x/len(self.train_loader) for x in running_ncc_per_frame]

        self.summary_writer.add_scalars("Losses/total_loss", {'train': epoch_total_loss}, self.current_epoch)
        self.summary_writer.add_scalars("Losses/similarity_loss", {'train': epoch_simi_loss}, self.current_epoch)
        self.summary_writer.add_scalars("Losses/smooth_loss", {'train': epoch_smooth_loss}, self.current_epoch)
        self.summary_writer.add_scalars("Losses/cyclic_loss", {'train': epoch_cyclic_loss}, self.current_epoch)
        for k in range(self.args.num_images_per_group):
            self.summary_writer.add_scalars(f"Frames_Reg/frame{k}", {'train': epoch_ncc_per_frame[k]}, self.current_epoch)


        self.logger.info( f'Training Reg, {self.current_epoch}, total loss {epoch_total_loss:.4f}, '
                          f'simi. loss {epoch_simi_loss:.4f}, smooth loss {epoch_smooth_loss:.4f}, '
                          f'cyclic loss {epoch_cyclic_loss:.4f}')


    def validate(self):

        # Set model to evaluation mode
        self.model.eval()
        # initialize stats
        running_total_loss = 0.
        running_simi_loss = 0.
        running_cyclic_loss = 0.
        running_smooth_loss = 0.
        running_ncc_per_frame = [0. for x in range(self.args.num_images_per_group)]
        i = 1

        with torch.no_grad():
            for val_batch in self.validation_loader:
                total_loss = 0.

                batch_init = val_batch['image'][tio.DATA].to(device)[0, ...]

                batch_resampled = F.interpolate(batch_init.unsqueeze(0).type(torch.float32), (self.args.image_shape[0],
                                                                                  self.args.image_shape[1],
                                                                                  self.args.num_images_per_group),
                                                    mode='trilinear', align_corners=True).squeeze(0)

                batch_mri = batch_resampled.permute(3, 0, 2, 1)  # (n,1,h,w)

                res = self.model(batch_mri)

                simi_loss, ncc_per_frame = self.ncc_loss(res['warped_input_image'], res['template'], device=device)
                simi_loss_item = simi_loss.item()
                total_loss += simi_loss

                if self.args.smooth_reg > 0:
                    if self.args.smooth_reg_type == 'dvf':
                        smooth_loss = model.loss.smooth_loss_simple(res['scaled_disp_t2i'],
                                                                    res['scaled_template'])
                    else:
                        smooth_loss = model.loss.smooth_loss(res['scaled_disp_t2i'],
                                                             res['scaled_template'])

                    total_loss += self.args.smooth_reg * smooth_loss
                    smooth_loss_item = smooth_loss.item()
                else:
                    smooth_loss_item = 0

                if self.args.cyclic_reg > 0:
                    cyclic_loss = (torch.mean((torch.sum(res['scaled_disp_t2i'], 0)) ** 2)) ** 0.5
                    total_loss += self.args.cyclic_reg * cyclic_loss
                    cyclic_loss_item = cyclic_loss.item()
                else:
                    cyclic_loss_item = 0

                running_ncc_per_frame = [x + y for x, y in zip(running_ncc_per_frame,
                                                               list(ncc_per_frame.cpu().detach().numpy()))]

                running_simi_loss += simi_loss_item
                running_smooth_loss += smooth_loss_item
                running_cyclic_loss += cyclic_loss_item

            total_loss_item = total_loss.item()
            running_total_loss += total_loss_item

            if self.current_epoch % self.args.save_image_every == 0:

                grid_input_img = torchvision.utils.make_grid(batch_mri, nrow=5)
                self.summary_writer.add_image(f'inputImage_{i}', grid_input_img, self.current_epoch)

                if 'disp_t2i' in res:
                    grid_warped_img = torchvision.utils.make_grid(res['warped_input_image'], nrow=5)
                    grid_temp = torchvision.utils.make_grid(res['template'], nrow=1)
                    self.summary_writer.add_image(f'warpedImage_{i}', grid_warped_img)
                    self.summary_writer.add_image(f'templateImage_{i}', grid_temp, self.current_epoch)

                i += 1

        epoch_total_loss = running_total_loss / len(self.validation_loader)
        epoch_simi_loss = running_simi_loss / len(self.validation_loader)
        epoch_smooth_loss = running_smooth_loss / len(self.validation_loader)
        epoch_cyclic_loss = running_cyclic_loss / len(self.validation_loader)
        epoch_ncc_per_frame = [x/len(self.validation_loader) for x in running_ncc_per_frame]

        self.summary_writer.add_scalars("Losses/total_loss", {'validation': epoch_total_loss}, self.current_epoch)
        self.summary_writer.add_scalars("Losses/similarity_loss", {'validation': epoch_simi_loss}, self.current_epoch)
        self.summary_writer.add_scalars("Losses/smooth_loss", {'validation': epoch_smooth_loss}, self.current_epoch)
        self.summary_writer.add_scalars("Losses/cyclic_loss", {'validation': epoch_cyclic_loss}, self.current_epoch)
        for k in range(self.args.num_images_per_group):
            self.summary_writer.add_scalars(f"Frames_Reg/frame{k}", {'train': epoch_ncc_per_frame[k]}, self.current_epoch)

        self.logger.info(f'Validation Reg, {self.current_epoch}, total loss {epoch_total_loss:.4f}, '
                         f' simi. loss {epoch_simi_loss:.4f}, smooth loss {epoch_smooth_loss:.4f}, '
                         f' cyclic loss {epoch_cyclic_loss:.3f}')


    def inference(self):

        if self.args.inference_set == 'validation':
            loader = self.validation_loader
        elif self.args.inference_set == 'test':
            loader = self.test_loader

        # switch model to evaluation mode
        self.model.eval()
        with torch.no_grad():
            for test_batch in loader:

                batch_init = test_batch['image'][tio.DATA].to(device)[0, ...]

                batch_resampled = F.interpolate(batch_init.unsqueeze(0), (self.args.image_shape[0],
                                                                                  self.args.image_shape[1],
                                                                                  self.args.num_images_per_group),
                                                    mode='trilinear', align_corners=True).squeeze(0)

                batch_mri = batch_resampled.permute(3, 0, 2, 1)  # (n,1,h,w)

                res = self.model(batch_mri)
                
                #TODO: Figure which version to use
                #patient_name = test_batch["mri"][tio.PATH][0].split('/')[-1].split('.mha')[0]

                patient_name = test_batch["image"][tio.PATH][0].split('/')[-1].split('.mha')[0]
                patient_output_path = os.path.join(self.args.output_dir, self.args.inference_set, patient_name)
                if not os.path.exists(patient_output_path):
                    os.makedirs(patient_output_path)

                copy_disp_t2i = res['disp_t2i'].clone().detach()
                copy_warped_input_image = res['warped_input_image'].clone().detach()
                copy_warped_input_image = copy_warped_input_image[:,0,:,:]
                batch_disp_t2i_resampled1 = F.interpolate(copy_disp_t2i[:, 0, ...].unsqueeze(1).permute(1, 2, 3, 0).unsqueeze(0),
                                                         (self.args.image_shape[0], self.args.image_shape[1],
                                                              batch_init.shape[-1]),
                                                         mode='trilinear').squeeze(0)
                batch_disp_t2i_resampled2 = F.interpolate(copy_disp_t2i[:, 1, ...].unsqueeze(1).permute(1, 2, 3, 0).unsqueeze(0),
                                                         (self.args.image_shape[0], self.args.image_shape[1],
                                                              batch_init.shape[-1]),
                                                         mode='trilinear').squeeze(0)

                batch_disp_t2i_resampled = torch.cat((batch_disp_t2i_resampled1, batch_disp_t2i_resampled2), dim=0)

                sitk_output_dvf = sitk.GetImageFromArray(batch_disp_t2i_resampled.permute(3, 1, 2, 0).cpu(), isVector=True)
                sitk_output_dvf.SetSpacing(sitk.ReadImage(test_batch["image"][tio.PATH][0]).GetSpacing())
                sitk_output_dvf.SetDirection(sitk.ReadImage(test_batch["image"][tio.PATH][0]).GetDirection())
                sitk.WriteImage(sitk_output_dvf, os.path.join(patient_output_path, 'dvf.mha'))
                
                '''
                copy_warped_input_image = sitk.GetImageFromArray(copy_warped_input_image)
                copy_warped_input_image.SetSpacing(sitk.ReadImage(test_batch["image"][tio.PATH][0]).GetSpacing())
                copy_warped_input_image.SetDirection(sitk.ReadImage(test_batch["image"][tio.PATH][0]).GetDirection())
                sitk.WriteImage(copy_warped_input_image, os.path.join(patient_output_path, 'wimage.mha'))
                '''
    


                print(f'finished patient {patient_name}')


    def eval(self):

        pass


    def finalize(self):
        """
        Finalize all the operations of the 2 Main classes of the process the operator and the data loader
        :return:
        """
        if self.args.debug or self.args.mode != 'train':
            pass
        else:
            self.logger.info("Please wait while finalizing the operation.. Thank you")
            self.summary_writer.export_scalars_to_json(os.path.join(self.args.tensorboard_dir, "all_scalars.json"))
            self.summary_writer.close()
