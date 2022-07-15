import matplotlib.pyplot as plt
import numpy as np

def plot_loss_landscape(min_curve_BN, max_curve_BN, min_curve, max_curve):
    plt.clf()
    x = list(range(len(min_curve)))
    x = np.array(x) 
    x_bn=list(range(len(min_curve_BN)))
    min_curve_BN = np.array(min_curve_BN)[:len(min_curve)] 
    max_curve_BN = np.array(max_curve_BN)[:len(min_curve)]
    min_curve = np.array(min_curve) 
    max_curve = np.array(max_curve) 
    
    # ax1 = plt.subplot(1, 2, 1, frameon = False)
    plt.plot(x, min_curve, color = 'royalblue', alpha=0.8)
    plt.plot(x, max_curve, color = 'royalblue', alpha=0.8)
    p1 = plt.fill_between(x, min_curve, max_curve, facecolor="royalblue", alpha=0.3)
    plt.title('Standard VGG')
    plt.ylabel('Gradient Difference')
    plt.xlabel('Step')
    
    plt.ylim((0, 3))
    
    # ax2 = plt.subplot(1, 2, 2, frameon = False)
    plt.plot(x, min_curve_BN, color = 'darkorange', alpha=0.8)
    plt.plot(x, max_curve_BN, color = 'darkorange', alpha=0.8)
    p2 = plt.fill_between(x, min_curve_BN, max_curve_BN, facecolor="darkorange", alpha=0.3)
    
    
    l1 = plt.legend([p1, p2], ["VGG_A", "VGG_A_BatchNorm"], loc='upper right')
    plt.gca().add_artist(l1)
    
    plt.title('"Effective" beta-smoothness')
    plt.ylabel('Gradient Difference')
    plt.xlabel('Step')
    
    plt.ylim((0, 3))
    plt.savefig("gradient difference.jpg")

def ReadFile(address):
    f = open(address, encoding='UTF-8')
    line = f.readline()
    ls = []
    while line:
        line_ = line.replace('\n','')
        line_ = line_.split('\t')
        line_ = line_[:-1]
        line_ = list(map(float,line_))
        ls = ls + line_
        line = f.readline()
    f.close()
    return ls

PATH='/home/newdisk/zxy/pj2/codes_for_pj/section2/draw/beta_smooth/'
min_curve_BN = ReadFile(PATH+'min_curve2_BN.txt') 
max_curve_BN = ReadFile(PATH+'max_curve2_BN.txt')
min_curve = ReadFile(PATH+'min_curve2.txt')
max_curve = ReadFile(PATH+'max_curve2.txt')
plot_loss_landscape(min_curve_BN, max_curve_BN, min_curve, max_curve)