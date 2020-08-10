from torch import nn, optim, cuda, backends
from utils import *
from models import *
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary

run = get_input()
## parameters
if run == -1:  # user-defined parameters
    # architecture
    model = 1 #1 is Gated PixelCNN
    filters = 3 # number of convolutional filters or feature maps
    layers = 5 # number of hidden convolutional layers
    bound_type = 0 # type of boundary layer for generation, 0 = empty
    boundary_layers = 2 # number of layers of conv_fields between output sample and boundary
    softmax_temperature = 0 # ratio to batch mean at which softmax will sample, set to 0 to sample at training temperature

    # training
    training_data = 2 # select training set: 1 - nanoparticle aggregate, 2 - MAC
    training_batch = 2048 # size of training and test batches - it will try to run at this size, but if it doesn't fit it will go smaller
    sample_batch_size = 2048  # max batch size for sample generator
    n_samples = 1  # total samples to be generated when we generate, must not be zero (it may make more if there is available memory)
    run_epochs = 1000 # number of incremental epochs which will be trained over - if zero, will run just the generator
    dataset_size = 400 # the maximum number of samples to consider from our dataset
    train_margin = 1e-2 # the convergence criteria for training error
    average_over = 5 # how many epochs to average over to determine convergence
    outpaint_ratio = 2 # sqrt of size of output relative to input
    generation_type = 2 # 1 - fast for many small images, 2 - fast for few, large images
    GPU = 1  # if 1, runs on GPU (requires CUDA), if 0, runs on CPU (slow! and may break some functions)
else: # when running on cluster, can take arguments from a batch_parameters script
    with open('batch_parameters.pkl', 'rb') as f:
        inputs = pickle.load(f)
    # architecture
    model = inputs['model'][run]
    filters = inputs['filters'][run]
    layers = inputs['layers'][run]
    bound_type = inputs['bound_type'][run]
    boundary_layers = inputs['boundary_layers'][run]
    softmax_temperature = inputs['softmax_temperature'][run]

    # training
    training_data = inputs['training_data'][run]
    training_batch = int(inputs['training_batch'][run])
    sample_batch_size = inputs['sample_batch_size'][run]
    n_samples = inputs['n_samples'][run]
    run_epochs = inputs['run_epochs'][run]
    dataset_size = inputs['dataset_size'][run]
    train_margin = inputs['train_margin'][run]
    average_over = int(inputs['average_over'][run])
    outpaint_ratio = inputs['outpaint_ratio'][run]
    generation_type = inputs['generation_type'][run]
    GPU = inputs['GPU'][run]

if GPU == 1:
    backends.cudnn.benchmark = True  # auto-optimizes certain backend processes

filter_size = 3  # initial layer kernel size MUST BE 3 IN CURRENT IMPLEMENTATION
dir_name = get_dir_name(model, training_data, filters, layers, filter_size, dataset_size)  # get directory name for I/O
writer = SummaryWriter('logfiles/'+dir_name[:]+'_T=%.3f'%softmax_temperature)  # initialize tensorboard writer

prev_epoch = 0
if __name__ == '__main__':  # run it!
    net, conv_field, optimizer, sample_0, input_x_dim, input_y_dim, sample_x_dim, sample_y_dim = initialize_training(model, filters, filter_size, layers, training_data, outpaint_ratio, dataset_size)
    net, optimizer, prev_epoch = load_checkpoint(net, optimizer, dir_name, GPU, prev_epoch)
    channels = sample_0.shape[1]
    out_maps = len(np.unique(sample_0)) + 1

    input_analysis = analyse_inputs(training_data, out_maps, dataset_size) # analyse inputs to prepare accuracy metrics

    if prev_epoch == 0: # if we are just beginning training, save inputs and relevant analysis
        save_outputs(dir_name, input_analysis, sample_0, softmax_temperature, 1)

    print('Imported and Analyzed Training Dataset {}'.format(training_data))

    if GPU == 1:
        net = nn.DataParallel(net) # go to multi-GPU training
        print("Using", torch.cuda.device_count(), "GPUs")
        net.to(torch.device("cuda:0"))
        #print(summary(net, (channels, input_x_dim, input_y_dim)))  # doesn't work on CPU, not sure why

    max_epochs = run_epochs + prev_epoch + 1

    ## BEGIN TRAINING/GENERATION
    if run_epochs == 0:  # no training, just samples
        prev_epoch += 1
        epoch = prev_epoch

        # to a test of the net to get it warmed up
        training_batch, changed = get_training_batch_size(training_data, training_batch, model, filters, filter_size, layers, out_maps, channels, dataset_size, GPU)  # confirm we can keep on at this batch size
        if changed == 1:  # if the training batch is different, we have to adjust our batch sizes and dataloaders
            tr, te = get_dataloaders(training_data, training_batch, dataset_size)
            print('Training batch set to {}'.format(training_batch))
        else:
            tr, te = get_dataloaders(training_data, training_batch, dataset_size)

        sample, time_ge, n_samples, agreements, output_analysis = generation(generation_type, dir_name, input_analysis, outpaint_ratio, epoch, model, filters, filter_size, layers, net, writer, te, out_maps, conv_field, sample_x_dim, sample_y_dim, n_samples, sample_batch_size, bound_type, training_data, boundary_layers, channels, softmax_temperature, dataset_size, GPU, cuda)

    else: #train it!
        epoch = prev_epoch + 1
        converged = 0
        tr_err_hist = []
        te_err_hist = []
        while (epoch <= (max_epochs + 1)) & (converged == 0):#for epoch in range(prev_epoch+1, max_epochs):  # over a certain number of epochs

            if (epoch - prev_epoch) < 3:
                training_batch, changed = get_training_batch_size(training_data, training_batch, model, filters, filter_size, layers, out_maps, channels, dataset_size, GPU)  # confirm we can keep on at this batch size
                if changed == 1: # if the training batch is different, we have to adjust our batch sizes and dataloaders
                    tr, te = get_dataloaders(training_data, training_batch, dataset_size)
                    print('Training batch set to {}'.format(training_batch))
                else:
                    tr, te = get_dataloaders(training_data, training_batch, dataset_size)

            err_tr, time_tr = train_net(net, optimizer, writer, tr, epoch, out_maps, GPU, cuda)  # train & compute loss
            err_te, time_te = test_net(net, writer, te, out_maps, epoch, GPU, cuda)  # compute loss on test set
            tr_err_hist.append(torch.mean(torch.stack(err_tr)))
            te_err_hist.append(torch.mean(torch.stack(err_te)))
            print('epoch={}; nll_tr={:.5f}; nll_te={:.5f}; time_tr={:.1f}s; time_te={:.1f}s'.format(epoch, torch.mean(torch.stack(err_tr)), torch.mean(torch.stack(err_te)), time_tr, time_te))

            save_ckpt(epoch, net, optimizer, dir_name[:]) #save checkpoint

            converged = auto_convergence(train_margin, average_over, epoch, prev_epoch, net, optimizer, dir_name, tr_err_hist, te_err_hist, max_epochs)

            epoch += 1

        tr, te = get_dataloaders(training_data, 4, 100)  # get something from the dataset
        example = next(iter(tr)).cuda()  # get seeds from test set
        raw_out = net(example[0:2, :, :, :].float())
        raw_out = F.softmax(raw_out, dim=1)
        raw_out = raw_out[0].unsqueeze(1)
        raw_grid = utils.make_grid(raw_out, nrow=int(out_maps), padding=0)
        raw_grid = raw_grid[0].cpu().detach().numpy()
        np.save('raw_outputs/' + dir_name[:], raw_grid)

        # generate samples
        sample, time_ge, n_samples, agreements, output_analysis = generation(generation_type, dir_name, input_analysis, outpaint_ratio, epoch, model, filters, filter_size, layers, net, writer, te, out_maps, conv_field, sample_x_dim, sample_y_dim, n_samples, sample_batch_size, bound_type, training_data, boundary_layers, channels, softmax_temperature, dataset_size, GPU, cuda)
