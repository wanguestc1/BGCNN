import torch
import torch.nn as nn
import argparse
import dataloader
import BGCNN
import BGCNNMT
import shutil
import time
import os

parse=argparse.ArgumentParser(description='BGCNN && BGCNN_MT')
parse.add_argument('--nclass',type=int,default=8)
parse.add_argument('--batchsize',type=int,default=64,help='set batchsize')
parse.add_argument('--momentum',type=float,default=0.99,help='set momentum')
parse.add_argument('--epochs',type=int,default='',help='set the number of training epochs')
parse.add_argument('--lr',type=float,default=0.001,help='set the learning rate for training')
parse.add_argument('--weight_decay',type=float,default=5e-4,help='set weight_decay for training')
parse.add_argument('--lr_patience',type=int,default='',help='set learning rate patience for training')
parse.add_argument('--gamma',type=float,default=0.1,help='set gamma for training')
parse.add_argument('--train_dir',type=str,default='./data/train',help='set dir of training data')
parse.add_argument('--model_root',type=str,default='./modelsave',help='set dir for saving model')
parse.add_argument('--print_freq',type=float,default=10,help='set print frequency for training')
parse.add_argument('--gpu',type=bool,default=True,help='set GPU for using or not')
args=parse.parse_args()
global best_prec1

######### load dataset
train_loader = dataloader.load_training(args.train_dir,args.batchsize)

######### load model
model = BGCNN.BGCNN_net(num_classes=8)
model.cuda()

######### optimizer
optimizer=torch.optim.SGD(model.parameters(),lr=args.lr,momentum=args.momentum,weight_decay=args.weight_decay)
lr_scheduler=torch.optim.lr_scheduler.StepLR(optimizer,args.lr_patience,gamma=args.gamma,last_epoch=-1)
criterion = nn.CrossEntropyLoss().cuda()

best_prec1 = 0





def train_net(train_loader, model, criterion, optimizer, epoch):
    """
        Run one train epoch
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for m, (input, lable) in enumerate(train_loader):
        if args.gpu:
            input = input.cuda()
            lable = lable.cuda()

        input = torch.autograd.Variable(input)
        lable = torch.autograd.Variable(lable)

        data_time.update(time.time() - end)

        # compute output
        output_cls= model(input)

        #loss
        loss_cls = criterion(output_cls, lable)

        loss =  loss_cls
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (m + 1) % args.print_freq == 0:
            print("Epoch: ", (epoch + 1), " Step: ", (m + 1), " /", total_step, " Loss: ", loss.item())



def validate(val_loader, model, criterion):
    """
    Run evaluation
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input, volatile=True).cuda()
        target_var = torch.autograd.Variable(target, volatile=True)


        # compute output
        output= model(input_var)
        loss = criterion(output, target_var)


        output = output.float()
        loss = loss.float()


        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      i, len(val_loader), batch_time=batch_time, loss=losses,
                      top1=top1))

    print(' * Prec@1 {top1.avg:.3f}'
          .format(top1=top1))

    return top1.avg

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """
    Save the training model
    """
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')
    pass

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res



for epoch in range(args.epochs):

    model = model.train()
    print("*****************" + str(epoch) + "*******************")
    lr_scheduler.step(epoch)
    total_step = len(train_loader)


    train_net(train_loader, model, criterion, optimizer, epoch)

    # evaluate
    prec1 = validate(train_loader, model, criterion)


    # remember best prec@1 and save checkpoint
    is_best = prec1 > best_prec1
    best_prec1 = max(prec1, best_prec1)
    #
    save_checkpoint({
        'state_dict': model.state_dict(),
        'best_prec1': best_prec1,
        'epoch': epoch,
    }, is_best, filename=os.path.join(args.model_root, 'model.th'))



















