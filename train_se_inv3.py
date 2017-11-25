import os
used_gpu = '0,1'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = used_gpu

from utils import *
from Trainer import *
from datetime import *
import torch.optim as optim
from tqdm import tqdm
from data_transform import *
from se_inception_v3 import SEInception3
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler, SequentialSampler
import torch.nn.functional as F
import time


cuda = torch.cuda.is_available()
total = 12371293
#total = 82
np.random.seed(2333)
to = np.arange(total)
to = np.random.permutation(to)
val_mask = to[:int(total*0.1)]
train_mask = to[int(total*0.1):]
print("finish mask")


def evaluate(net, test_loader):
    test_num = 0
    test_loss = 0
    test_acc = 0
    pbar = tqdm(total=len(test_loader))
    for images, labels in test_loader:
        images = Variable(images, volatile=True).cuda()
        labels = Variable(labels).cuda()

        logits = net(images)
        probs = F.softmax(logits)
        loss = F.cross_entropy(logits, labels)
        acc = (probs.data.max(1)[1] == labels.data).sum()

        batch_size = 256
        test_acc += acc
        test_loss += batch_size * loss.data[0]
        test_num += batch_size
        pbar.update()
    pbar.close()
    test_acc = test_acc / test_num
    test_loss = test_loss / test_num
    return test_loss, test_acc


def train():
    out_dir = os.getcwd() + "/results"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    if not os.path.exists(out_dir+'/checkpoint'):
        os.makedirs(out_dir+"/checkpoint")

    initial_checkpoint = None
    data_path = "/data/lixiang/train.bson"
    #data_path = "/data/lixiang/train_example.bson"
    pretrained_model = "epoch-5-acc-0.67.pth"

    logging.basicConfig(level=logging.DEBUG,
                        format='%(message)s%(asctime)s',
                        datefmt='%d %b %H:%M:%S',
                        filename='trainlog.log',
                        filemode='a')
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s %(message)s', datefmt='%d %b %H:%M:%S')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

    logging.debug('\t--- [START %s] %s' % (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '-' * 64))
    logging.debug('** some experiment setting **')
    logging.debug('\tout_dir      = %s' % out_dir)
    logging.debug('\tsaved model    = %s' % pretrained_model)
    logging.debug('\tinitial_checkpoint = %s' % initial_checkpoint)

    logging.debug('** net setting **')
    net = SEInception3(in_shape=(3, 224, 224), num_classes=5270)
    logging.debug('%s'%(type(net)))
    logging.debug('%s'%(str(net)))

    if cuda:
        net.cuda()
    if len(used_gpu) > 1 and cuda:
        net = torch.nn.DataParallel(net, device_ids=[0,1])

    num_workers = 5
    num_iters = 1000 * 1000
    iter_smooth = 20
    iter_log = 1000
    iter_valid = 500
    iter_save = 1000

    optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()),
                          lr=0.05, momentum=0.9, weight_decay=0.0001)
    lr_step = MultiStepLR(optimizer, [4, 8])

    logging.debug('** dataset setting **')
    batch_size = 256
    iter_accum = 1
    data_frame = extract_categories_df(data_path)
    train_dataset = CdiscountTrain(data_path, data_frame, train_mask, transform=train_augment)
    train_loader = DataLoader(train_dataset,
                              sampler=RandomSampler(train_dataset),
                              batch_size=batch_size,
                              drop_last=True,
                              num_workers=num_workers)

    valid_dataset = CdiscountVal(data_path, data_frame, val_mask, transform=valid_augment)
    valid_loader = DataLoader(valid_dataset,
                              sampler=SequentialSampler(valid_dataset),
                              batch_size=batch_size,
                              drop_last=False,
                              num_workers=num_workers)

    #logging.debug('\ttrain_dataset.split = %s'%train_dataset.mask)
    #logging.debug('\tvalid_dataset.split = %s'%valid_dataset.mask)
    logging.debug('\tlen(train_dataset)  = %d'%(len(train_dataset)))
    logging.debug('\tlen(valid_dataset)  = %d'%(len(valid_dataset)))
    logging.debug('\tlen(train_loader)   = %d'%(len(train_loader)))
    logging.debug('\tlen(valid_loadernum_iters) = %d'%(len(valid_loader)))
    logging.debug('\tbatch_size  = %d'%batch_size)
    logging.debug('\titer_accum  = %d'%iter_accum)
    logging.debug('\tbatch_size*iter_accum  = %d\n'%batch_size*iter_accum)
    logging.debug('\n')

    start_iter = 0
    start_epoch = 0.
    if initial_checkpoint is not None:
        if len(used_gpu) > 1 and cuda:
            net.module.load_state_dict(torch.load(initial_checkpoint, map_location=lambda storage, loc: storage))
        else:
            net.load_state_dict(torch.load(initial_checkpoint, map_location=lambda storage, loc: storage))
        checkpoint = torch.load(initial_checkpoint.replace('_model.pth', '_optimizer.pth'),
                                map_location=lambda storage, loc: storage)
        start_iter = checkpoint['iter']
        start_epoch = checkpoint['epoch']
        optimizer.load_state_dict(checkpoint['optimizer'])
    elif pretrained_model is not None:
        if len(used_gpu) > 1 and cuda:
            net.module.load_pretrained_model('saved_models/' + pretrained_model)
        else:
            net.load_pretrained_model('saved_models/' + pretrained_model)

    logging.debug('** start training here! **')
    logging.debug(' optimizer=%s' % str(optimizer))
    logging.debug(' LR=%s' % str(lr_step))
    logging.debug('   rate   iter   epoch  | valid_loss/acc | train_loss/acc | batch_loss/acc |  time   ')
    logging.debug('-------------------------------------------------------------------------------------')

    train_loss = 0.0
    train_acc = 0.0
    valid_loss = 0.0
    valid_acc = 0.0
    batch_loss = 0.0
    batch_acc = 0.0
    rate = 0

    start = time.time()
    j = 0
    i = 0

    while i < num_iters:  # loop over the dataset multiple times
        sum_train_loss = 0.0
        sum_train_acc = 0.0
        sum_ = 0

        net.train()
        optimizer.zero_grad()
        for images, labels in train_loader:
            i = j / iter_accum + start_iter
            epoch = (i - start_iter) * batch_size * iter_accum / len(train_dataset) + start_epoch

            if i % iter_valid == 0:
                net.eval()
                valid_loss, valid_acc = evaluate(net, valid_loader)
                net.train()

            if i % iter_log == 0:
                print('\r', end='', flush=True)
                logging.debug('%0.4f  %5.1f k   %4.2f  | %0.4f  %0.4f | %0.4f  %0.4f | %0.4f  %0.4f | %5.0f min \n' % \
                              (rate, i / 1000, epoch, valid_loss, valid_acc, train_loss, train_acc, batch_loss, batch_acc,
                              (time.time() - start) / 60))

            if j % iter_save == 0:
                torch.save(net.state_dict(), out_dir + '/checkpoint/%08d_model.pth' % i)
                torch.save({'optimizer': optimizer.state_dict(),
                            'iter': i,'epoch': epoch},
                            out_dir + '/checkpoint/%08d_optimizer.pth' % i)

            # learning rate schduler
            lr = lr_step.get_lr()[0]
            rate = lr * iter_accum

            # one iteration update
            images = Variable(images).cuda()
            labels = Variable(labels).cuda()
            logits = net(images)
            loss = F.cross_entropy(logits, labels)
            acc = (logits.data.max(1)[1] == labels.data).sum()/batch_size

            # one-time gradient
            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()

            # accumulate gradients
            loss.backward()
            if j % iter_accum == 0:
                # torch.nn.utils.clip_grad_norm(net.parameters(), 1)
                optimizer.step()
                optimizer.zero_grad()

            # print statistics
            batch_acc = acc
            batch_loss = loss.data[0]
            sum_train_loss += batch_loss
            sum_train_acc += batch_acc
            sum_ += 1
            if i % iter_smooth == 0:
                train_loss = sum_train_loss / sum_
                train_acc = sum_train_acc / sum_
                sum_train_loss = 0.
                sum_train_acc = 0.
                sum_ = 0

            print('\r%0.4f  %5.1f k   %4.2f  | %0.4f  %0.4f | %0.4f  %0.4f | %0.4f  %0.4f | %5.0f min  %d, %d' %\
                  (rate, i / 1000, epoch, valid_loss, valid_acc, train_loss, train_acc, batch_loss, batch_acc,
                   (time.time() - start)/60, int(i), int(j)), end='', flush=True)

            j = j + 1
        pass  # end of one data loader
    pass

    # save model
    torch.save(net.state_dict(), out_dir + '/checkpoint/%d_model.pth' % i)
    torch.save({'optimizer': optimizer.state_dict(),
                'iter': i, 'epoch': epoch},
               out_dir + '/checkpoint/%d_optimizer.pth' % i)
    logging.debug("** Training Finishs! **")


if __name__ == "__main__":
    train()