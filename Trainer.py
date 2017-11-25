import os
import torch
import logging
from torch.autograd import Variable


class Trainer(object):
    cuda = torch.cuda.is_available()

    def __init__(self, model, optimizer, loss_f, batch_size, distrit=False, save_freq=1, print_freq=10, val_freq=500):
        self.distrit = distrit
        self.model = model.module if self.distrit else model
        if self.cuda:
            model.cuda()
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.loss_f =loss_f
        save_dir = os.getcwd() + "/saved_models"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.save_dir = save_dir
        self.save_freq = save_freq
        self.print_freq = print_freq
        self.val_freq = val_freq

        # Information Saving and Printing
        logging.basicConfig(level=logging.DEBUG,
                            format='%(asctime)s %(message)s',
                            datefmt='%d %b %H:%M:%S',
                            filename='train.log',
                            filemode='a')
        console = logging.StreamHandler()
        console.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s %(message)s', datefmt='%d %b %H:%M:%S')
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)

    def _loop(self, data_loader, epoch, is_train=True):
        loop_loss = []
        correct = []
        i = 0
        for data, target in data_loader:
            i += 1
            if self.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data, volatile=not is_train), Variable(target, volatile=not is_train)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.loss_f(output, target)
            loop_loss.append(loss.data[0] / len(data_loader))
            acc = (output.data.max(1)[1] == target.data).sum()
            correct.append(acc / len(data_loader.dataset))
            if is_train:
                loss.backward()
                self.optimizer.step()
                if i % self.print_freq == 0:
                    logging.info('{} ep: {: >4d}th/{:<4d} loss: {:>5.2f}'
                                 '/accuracy: {:>5.2%}'.format(epoch, i,
                                                              len(data_loader),
                                                              loss.data[0],
                                                              acc / self.batch_size))
        mode = "train" if is_train else "test"
        logging.warning(">>>[{: >5s}] loss: {:.2f}/accuracy: {:.2%}".format(mode, sum(loop_loss), sum(correct)))
        return loop_loss, correct

    def train(self, data_loader, epoch):
        self.model.train()
        loss, correct = self._loop(data_loader, epoch)
        return loss, correct

    def test(self, data_loader, epoch):
        self.model.eval()
        loss, correct = self._loop(data_loader, epoch, is_train=False)
        return loss, sum(correct)

    def loop(self, train_data, test_data, scheduler=None):
        ep = 0
        while True:
            logging.info("epochs: {}".format(ep) + '\n')
            self.train(train_data, ep)
            _, test_acc = self.test(test_data, ep)
            if scheduler is not None:
                scheduler.step(epoch=ep)
            ep += 1
            if ep % self.save_freq == 0:
                self.save(ep, test_acc)

    def save(self, epoch, test_acc):
        prefix = "epoch-{}".format(epoch)
        model_name = prefix + "-acc-{:.4f}".format(test_acc) + "-model.pth"
        torch.save(self.model.state_dict(),
                   os.path.join(self.save_dir, model_name))
        torch.save(self.optimizer.state_dict(),
                   os.path.join(self.save_dir, prefix + "-opt.pth"))
        logging.debug(">>>[ save] {:d} th".format(int(epoch // self.save_freq)) + '\n')