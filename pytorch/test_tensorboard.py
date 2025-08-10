from torch.utils.tensorboard import SummaryWriter


'''
使用方法： tensorboard --logdir=logs --port=6007 --bind_all
'''

if __name__ == '__main__':
    sw = SummaryWriter("logs")

    for i in range(0 , 100):
        sw.add_scalar("y=x" , i , i )
    sw . close()