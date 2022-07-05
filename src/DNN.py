# coding=utf-8
import torch
import numpy as np
import matplotlib.pyplot as plt
class Net(torch.nn.Module):
    def __init__(self, NL, NN,device):
        # NL-隐藏层
        # NN-神经元数量
        self.device = device
        super(Net, self).__init__()
        self.input_layer = torch.nn.Linear(1, NN).to(self.device)
        self.hidden_layer = torch.nn.ModuleList([torch.nn.Linear(NN, NN) for i in range(NL)]).to(self.device)
        self.output_layer = torch.nn.Linear(NN, 1).to(self.device)
    def forward(self, x):
        o = self.act(self.input_layer(x))
        for i, li in enumerate(self.hidden_layer):
            o = self.act(li(o))
        out = self.output_layer(o)
        return out
    def act(self, x):
        return torch.tanh(x)

def updateGraph(x,y,loss,out,f,exact_flag):
  plt.cla()   # 清除当前图形中的当前活动轴。其他轴不受影响
  if(exact_flag):
    plt.scatter(x.cpu().detach().numpy(), y.cpu().detach().numpy())  # 打印原始数据
    plt.text(0.5, 2, 'solution=%s' % f, fontdict={'size': 15, 'color': 'red'})  # 打印解形式
  plt.plot(x.cpu().detach().numpy(), out.cpu().detach().numpy(), 'r-', lw=2)  # 打印预测数据
  plt.text(0.5, 0, 'Loss=%.6f' % loss, fontdict={'size': 15, 'color': 'red'})  # 打印误差值
  plt.pause(0.05)

def finalGraph(x,y,loss,out,f,exact_flag):
  plt.cla()   # 清除当前图形中的当前活动轴。其他轴不受影响
  if(exact_flag):
    plt.scatter(x.cpu().detach().numpy(), y.cpu().detach().numpy())  # 打印原始数据
    plt.text(0.5, 2, 'solution=%s' % f, fontdict={'size': 15, 'color': 'red'})  # 打印解形式
  plt.plot(x.cpu().detach().numpy(), out.cpu().detach().numpy(), 'r-', lw=2)  # 打印预测数据
  plt.text(0.5, 0, 'Loss=%.6f' % loss, fontdict={'size': 15, 'color': 'red'})  # 打印误差值
  plt.ioff()
  plt.show()
  plt.pause(0)

def updateGraphSystem(t,x,y,z,loss,out1,out2,out3,f,exact_flag):
  plt.cla()   # 清除当前图形中的当前活动轴。其他轴不受影响
  if(exact_flag):
    plt.scatter(t.cpu().detach().numpy(), x.cpu().detach().numpy())  # 打印原始数据
    plt.scatter(t.cpu().detach().numpy(), y.cpu().detach().numpy())  # 打印原始数据
    if z!= None:plt.scatter(t.cpu().detach().numpy(), z.cpu().detach().numpy())  # 打印原始数据
    plt.text(0.5, 2, 'solution=%s' % f, fontdict={'size': 15, 'color': 'red'})  # 打印解形式
  plt.plot(t.cpu().detach().numpy(), out1.cpu().detach().numpy(), 'r-', lw=2)  # 打印预测数据
  plt.plot(t.cpu().detach().numpy(), out2.cpu().detach().numpy(), 'g-', lw=2)  # 打印预测数据
  if out3 != None: plt.plot(t.cpu().detach().numpy(), out3.cpu().detach().numpy(), 'b-', lw=2)  # 打印预测数据
  plt.text(0.5, 0, 'Loss=%.6f' % loss, fontdict={'size': 15, 'color': 'red'})  # 打印误差值
  plt.pause(0.05)

def finalGraphSystem(t,x,y,z,loss,out1,out2,out3,f,exact_flag):
  plt.cla()   # 清除当前图形中的当前活动轴。其他轴不受影响
  if(exact_flag):
    plt.scatter(t.cpu().detach().numpy(), x.cpu().detach().numpy())  # 打印原始数据
    plt.scatter(t.cpu().detach().numpy(), y.cpu().detach().numpy())  # 打印原始数据
    if z!= None:plt.scatter(t.cpu().detach().numpy(), z.cpu().detach().numpy())  # 打印原始数据
    plt.text(0.5, 2, 'solution=%s' % f, fontdict={'size': 15, 'color': 'red'})  # 打印解形式
  plt.plot(t.cpu().detach().numpy(), out1.cpu().detach().numpy(), 'r-', lw=2)  # 打印预测数据
  plt.plot(t.cpu().detach().numpy(), out2.cpu().detach().numpy(), 'g-', lw=2)  # 打印预测数据
  if out3 != None: plt.plot(t.cpu().detach().numpy(), out3.cpu().detach().numpy(), 'b-', lw=2)  # 打印预测数据
  plt.text(0.5, 0, 'Loss=%.6f' % loss, fontdict={'size': 15, 'color': 'red'})  # 打印误差值
  plt.ioff()
  plt.show()
  plt.pause(0)