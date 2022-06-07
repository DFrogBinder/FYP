import os
from pathlib import Path
import torchio as tio
import torch
import matplotlib as plt

def data_loader(train_data_folder=None, validation_data_folder=None, test_data_folder=None, debug_data_folder=None,
                num_workers=1, aug_type='aug0'):

    if train_data_folder is not None:

        train_images_dir = Path(os.path.join(train_data_folder, 'images'))
        train_image_paths = sorted(train_images_dir.glob('*.mha'))

        print('All train image paths are: ')
        for path in train_image_paths:
            print(path)

        train_subjects = []
        for image_path in train_image_paths:
            subject = tio.Subject(
                image=tio.ScalarImage(image_path),
            )
            train_subjects.append(subject)

        # z,W,H,N=train_subjects[0].image.data.shape
        # for i,subject in zip(range(len(train_subjects)), train_subjects):
        #     train_subjects[i].image.data=train_subjects[i].image.data.reshape(N,z,W,H)
        
        print("Shape of training image before loading is: "+str(train_subjects[0].image.data.shape))
        
        if aug_type == 'aug0':
            training_transform = tio.Compose([])
        elif aug_type == 'aug1':
            training_transform = tio.Compose([tio.RandomFlip(axes=(0, 1, 2))])
        elif aug_type == 'aug2':
            training_transform = tio.Compose([tio.RandomFlip(axes=(0, 1, 2)),
                                              tio.RandomNoise(mean=0, std=0.1),
                                              tio.RandomBlur(std=(2.5, 2.5, 0.0))])
        elif aug_type == 'aug4':
            training_transform = tio.Compose([tio.RandomAffine(degrees=0, scales=(0.15, 0.15, 0), translation=(40, 40, 0),
                                                               default_pad_value='minimum', image_interpolation='linear'),
                                              tio.RandomFlip(axes=(0, 1, 2)),
                                              tio.RandomNoise(mean=0, std=0.1),
                                              tio.RandomBlur(std=(2.5, 2.5, 0.0))])
        elif aug_type == 'aug5':
            training_transform = tio.Compose([tio.RandomFlip(axes=(0, 1, 2)),
                                              tio.RandomAffine(degrees=(0, 0, 0, 0, -10, 10),
                                                               scales=0,
                                                               translation=0,
                                                               center='image',
                                                               default_pad_value='minimum',
                                                               image_interpolation='linear')])

        train_set = tio.SubjectsDataset(train_subjects, transform=training_transform)
        print('Current transform for training data '+ str(training_transform))

        # # Plotting the first patient for inspection
        # print("Plotting first subject from the train set...")
        # Single_Subject = train_set[0]
        # Single_Subject.plot()


        print('Training set:', len(train_set), 'subjects')
        print('Train set is a ' + str(type(train_set)))
        print('The first entity of training set is '+ str(type(train_set[0])))
        print('The length of a single subject is '+ str(len(train_set[0])))
        train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=1,
            shuffle=True,
            num_workers=num_workers)

    else:
        train_loader = None

    if validation_data_folder is not None:

        validation_images_dir = Path(os.path.join(validation_data_folder, 'images'))
        validation_image_paths = sorted(validation_images_dir.glob('*.mha'))

        validation_subjects = []
        for image_path in validation_image_paths:
            subject = tio.Subject(
                image=tio.ScalarImage(image_path),
            )
            validation_subjects.append(subject)

        validation_transform = tio.Compose([])

        validation_set = tio.SubjectsDataset(validation_subjects, transform=validation_transform)

        print('Validation set:', len(validation_set), 'subjects')

        validation_loader = torch.utils.data.DataLoader(
            validation_set,
            batch_size=1,
            num_workers=num_workers)

    else:
        validation_loader = None

    if test_data_folder is not None:

        test_images_dir = Path(os.path.join(test_data_folder, 'images'))
        test_image_paths = sorted(test_images_dir.glob('*.mha'))

        test_subjects = []
        for image_path in test_image_paths:
            subject = tio.Subject(
                image=tio.ScalarImage(image_path),
            )
            test_subjects.append(subject)

        test_transform = tio.Compose([])

        test_set = tio.SubjectsDataset(test_subjects, transform=test_transform)

        print('Test set:', len(test_set), 'subjects')



        test_loader = torch.utils.data.DataLoader(
            test_set,
            batch_size=1,
            num_workers=num_workers)

    else:
        test_loader = None

    if debug_data_folder is not None:

        debug_images_dir = Path(os.path.join(debug_data_folder, 'images'))
        debug_image_paths = [sorted(debug_images_dir.glob('*.mha'))[0]]

        debug_subjects = []
        for image_path in debug_image_paths:
            subject = tio.Subject(
                image=tio.ScalarImage(image_path),
            )
            debug_subjects.append(subject)
        
        # z,W,H,N=debug_subjects[0].image.data.shape
        # for i,subject in zip(range(len(debug_subjects)), debug_subjects):
        #     debug_subjects[i].image.data=debug_subjects[i].image.data.reshape(N,z,H,W)
        
        print("Shape of debug image before loading is: "+str(debug_subjects[0].image.data.shape))

        if aug_type == 'aug0':
            debug_transform = tio.Compose([])
        elif aug_type == 'aug1':
            debug_transform = tio.Compose([tio.RandomFlip(axes=(0, 1, 2))])
        elif aug_type == 'aug2':
            debug_transform = tio.Compose([tio.RandomFlip(axes=(0, 1, 2)),
                                              tio.RandomNoise(mean=0, std=0.1),
                                              tio.RandomBlur(std=(2.5, 2.5, 0.0))])
        elif aug_type == 'aug4':
            debug_transform = tio.Compose([tio.RandomAffine(degrees=0, scales=(0.15, 0.15, 0), translation=(40, 40, 0),
                                                               default_pad_value='minimum', image_interpolation='linear'),
                                              tio.RandomFlip(axes=(0, 1, 2)),
                                              tio.RandomNoise(mean=0, std=0.1),
                                              tio.RandomBlur(std=(2.5, 2.5, 0.0))])
        elif aug_type == 'aug5':
            debug_transform = tio.Compose([tio.RandomFlip(axes=(0, 1, 2)),
                                              tio.RandomAffine(degrees=(0, 0, 0, 0, -10, 10),
                                                               scales=0,
                                                               translation=0,
                                                               center='image',
                                                               default_pad_value='minimum',
                                                               image_interpolation='linear')])


        debug_set = tio.SubjectsDataset(debug_subjects, transform=debug_transform)

        # Plotting the first patient for inspection
        # print("Plotting first subject from the debug set...")
        # Single_Subject = debug_set[0]
        # Single_Subject.plot()

        print('Debug set:', len(debug_set), 'subjects')



        debug_loader = torch.utils.data.DataLoader(
            debug_set,
            batch_size=1,
            num_workers=num_workers)

    else:
        debug_loader = None

    return train_loader, validation_loader, test_loader, debug_loader