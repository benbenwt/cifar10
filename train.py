# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# --datadir 1 --name cifar10 --dataset cifar10  --model BiT-M-R50x1 --logdir ./log --eval_every 10
# Lint as: python3
"""Fine-tune a BiT model on some downstream dataset."""
#!/usr/bin/env python3
# coding: utf-8
from os.path import join as pjoin  # pylint: disable=g-importing-member
import time

import numpy as np
import torch
import torchvision as tv

import models as models


import bit_hyperrule

from sklearn.metrics import f1_score, precision_score, recall_score



def topk(output, target, ks=(1,)):
  # 每行（每个样本）挑最大的五个列，512*10->512*5,512为设置的batch_size
  _, pred = output.topk(max(ks), 1, True, True)
  #转置矩阵 512*5->5*512
  pred = pred.t()
  _,one_pred=output.topk(max((1,)),1,True,True)
  one_pred=one_pred.t()
  classes = ('plane', 'car', 'bird', 'cat',
             'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
  # print("predict:",[classes[one_pred[0][j]] for j in range(4)])
  # target，1*512->5*512
  # print('target_init',target.size())
  # print('target:',target.view(1, -1).expand_as(pred).size())
  correct = pred.eq(target.view(1, -1).expand_as(pred))
  # print('correct:',correct.size())
  # 每列的最大值，[0]是bool取值，[1]是最大值所在行索引.一列为一张图片，从中挑选最大？不是为true嘛。
  # print('correct',[correct[:k].max(0)[0] for k in ks])
  result=[correct[:k].max(0)[0] for k in ks]
  result.append([one_pred[0][j] for j in range(4)])
  return  result

def recycle(iterable):
  while True:
    for i in iterable:
      yield i


def mktrainval():
  precrop, crop = bit_hyperrule.get_resolution_from_dataset("cifar10")
  train_tx = tv.transforms.Compose([
      tv.transforms.Resize((precrop, precrop)),
      tv.transforms.RandomCrop((crop, crop)),
      tv.transforms.RandomHorizontalFlip(),
      tv.transforms.ToTensor(),
      tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
  ])
  val_tx = tv.transforms.Compose([
      tv.transforms.Resize((crop, crop)),
      tv.transforms.ToTensor(),
      tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
  ])

  train_set = tv.datasets.ImageFolder(root=r'train', transform=train_tx)
  valid_set = tv.datasets.ImageFolder(root=r'test', transform=val_tx)


  # if args.examples_per_class is not None:
  #
  #   indices = fs.find_fewshot_indices(train_set, args.examples_per_class)
  #   train_set = torch.utils.data.Subset(train_set, indices=indices)


  batch_size = 600

  valid_loader = torch.utils.data.DataLoader(
      valid_set, batch_size=4, shuffle=True,
      num_workers=2, pin_memory=True, drop_last=False)

  if batch_size <= len(train_set):
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=2, pin_memory=True, drop_last=False)
  else:
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, num_workers=2, pin_memory=True,
        sampler=torch.utils.data.RandomSampler(train_set, replacement=True, num_samples=512))

  return train_set, valid_set, train_loader, valid_loader


def run_eval(model, data_loader, device, step):

  model.eval()

  all_c, all_top1, all_top5 = [], [], []
  end = time.time()
  y_true=[]
  y_pred=[]
  for b, (x, y) in enumerate(data_loader):
    with torch.no_grad():
      x = x.to(device, non_blocking=True)
      y = y.to(device, non_blocking=True)

      y_true.extend(y)
      classes = ('plane', 'car', 'bird', 'cat',
                 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
      # print("truth: ",' '.join('%5s' % classes[y[j]] for j in range(4)))
      logits = model(x)

      c = torch.nn.CrossEntropyLoss(reduction='none')(logits, y)
      top1, top5,y_pred_tmp = topk(logits, y, ks=(1, 5))
      y_pred.extend(y_pred_tmp)
      all_c.extend(c.cpu())  # Also ensures a sync point.
      all_top1.extend(top1.cpu())
      all_top5.extend(top5.cpu())
  model.train()

  print(f"validation loss {np.mean(all_c):.5f}, "
              f",validation accu {np.mean(all_top1):.2%}")

  f1 = f1_score(y_true, y_pred, average='micro')
  p = precision_score(y_true, y_pred, average='micro')
  r = recall_score(y_true, y_pred, average='micro')

  print("f1="+f1, "precision="+p, "recall="+r)
  return np.mean(all_c)

import matplotlib.pyplot as plt

def imshow(img):
    print('img:',img.size())
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def mixup_data(x, y, l):
  indices = torch.randperm(x.shape[0]).to(x.device)

  mixed_x = l * x + (1 - l) * x[indices]
  y_a, y_b = y, y[indices]
  return mixed_x, y_a, y_b


def mixup_criterion(criterion, pred, y_a, y_b, l):
  return l * criterion(pred, y_a) + (1 - l) * criterion(pred, y_b)


def main():

  torch.backends.cudnn.benchmark = True

  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  print(device)

  train_set, valid_set, train_loader, valid_loader = mktrainval()

  model = models.KNOWN_MODELS["BiT-M-R50x1"](head_size=len(valid_set.classes), zero_head=True)
  model.load_from(np.load(f"BiT-M-R50x1.npz"))

  model = torch.nn.DataParallel(model)
  step = 0

  optim = torch.optim.SGD(model.parameters(), lr=0.003, momentum=0.9)

  savename = pjoin("./log", "cifar10", "bit.pth.tar")
  try:
    checkpoint = torch.load(savename, map_location="cpu")
    step = checkpoint["step"]
    model.load_state_dict(checkpoint["model"])
    optim.load_state_dict(checkpoint["optim"])

  except FileNotFoundError:
    print('model not fount')
  model = model.to(device)
  optim.zero_grad()

  model.train()
  mixup = bit_hyperrule.get_mixup(len(train_set))
  cri = torch.nn.CrossEntropyLoss().to(device)

  mixup_l = np.random.beta(mixup, mixup) if mixup > 0 else 1
  all_top1 = []
  all_loss=[]
  all_val_loss=[]
  for x, y in  recycle(train_loader):
    x = x.to(device, non_blocking=True)
    y = y.to(device, non_blocking=True)

    lr = bit_hyperrule.get_lr(step, len(train_set),0.003)
    if lr is None:
      break
    for param_group in optim.param_groups:
      param_group["lr"] = lr

    if mixup > 0.0:
      x, y_a, y_b = mixup_data(x, y, mixup_l)


    logits = model(x)
    if mixup > 0.0:
      c = mixup_criterion(cri, logits, y_a, y_b, mixup_l)
    else:
      c = cri(logits, y)
    c_num = float(c.data.cpu().numpy())

    c.backward()
    top1, _,_1= topk(logits, y, ks=(1, 5))
    all_top1.extend(top1)
    print(f"[step {step}]: loss={c_num:.5f} ,accu={np.mean(all_top1):.2%} (lr={lr:.1e})")
    all_loss.append(c_num)
    all_top1=[]
    optim.step()
    optim.zero_grad()
    step += 1

    mixup_l = np.random.beta(mixup, mixup) if mixup > 0 else 1

    val_loss = run_eval(model, valid_loader, device, step)
    all_val_loss.append(val_loss)

    if  step%10 == 0:
      torch.save({
        "step": step,
        "model": model.state_dict(),
        "optim": optim.state_dict(),
      }, savename)
      print("model save to" + savename)
  plt.figure(figsize=(8, 8))
  plt.plot(range(1,11), all_loss, label='Training loss')
  plt.plot(range(1,11),all_val_loss,label='Validation loss')
  plt.legend(loc='lower right')
  plt.title(' loss and step')
  plt.show()


if __name__ == "__main__":
  main()