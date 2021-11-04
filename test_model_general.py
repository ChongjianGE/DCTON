import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
#from util import html
import time


if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    opt.num_threads = 0   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    model.setup(opt)
    res = []
    for i, data in enumerate(dataset):
        start = time.time()
        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break
        model.set_input(data)  # unpack data from data loader
        name = data['name']
        name = str(name).split('\'')[1].split('_')[0]+'_0'
        model.test(name)           # run inference
        end = time.time()
        res.append(end - start)
        if i % 50 == 0:  # save images to an HTML file
            print('processing (%04d)-th image... ' % (i))
    print('Finish the processing')


