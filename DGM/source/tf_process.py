import os, math

import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
# import psutil
# from os import getpid


def make_dir(path):

    try: os.mkdir(path)
    except: pass

def gray2rgb(gray):

    rgb = np.ones((gray.shape[0], gray.shape[1], 3)).astype(np.float32)
    rgb[:, :, 0] = gray[:, :, 0]
    rgb[:, :, 1] = gray[:, :, 0]
    rgb[:, :, 2] = gray[:, :, 0]

    return rgb

def dat2canvas(data):

    numd = math.ceil(np.sqrt(data.shape[0]))
    [dn, dh, dw, dc] = data.shape
    canvas = np.ones((dh*numd, dw*numd, dc)).astype(np.float32)

    for y in range(numd):
        for x in range(numd):
            try: tmp = data[x+(y*numd)]
            except: pass
            else:
                canvas[(y*dh):(y*dh)+224, (x*dw):(x*dw)+224, :] = (tmp+1)/2
    if(dc == 1):
        canvas = gray2rgb(gray=canvas)

    return canvas

def save_img(contents, names=["", "", ""], savename=""):

    num_cont = len(contents)
    plt.figure(figsize=(5*num_cont+2, 5))

    for i in range(num_cont):
        plt.subplot(1,num_cont,i+1)
        plt.title(names[i])
        plt.imshow(dat2canvas(data=contents[i]))

    plt.tight_layout()
    plt.savefig(savename)
    plt.close()

def boxplot(contents, savename=""):

    data, label = [], []
    for cidx, content in enumerate(contents):
        data.append(content)
        label.append("class-%d" %(cidx))

    # plt.clf()
    # fig, ax1 = plt.subplots()
    # bp = ax1.boxplot(data, showfliers=True, whis=3)
    # ax1.set_xticklabels(label, rotation=45)

    # plt.tight_layout()
    plt.savefig(savename)
    # plt.close()

def training(neuralnet, dataset, epochs, batch_size, normalize=True, load=False):

    print("\nTraining to %d epochs (%d of minibatch size)" %(epochs, batch_size))

    # print("Restoring model....")
    if load:
      neuralnet.load_parameter(model='model_checker')

    make_dir(path="training")
    result_list = ["restoration"]
    for result_name in result_list: make_dir(path=os.path.join("training", result_name))

    iteration = 0
    test_sq = 5
    test_size = test_sq**2
    
    for epoch in range(epochs):
        
        x_tr, y_tr, _, x_normals = dataset.next_train(batch_size=test_size, fix=True) # Initial batch
        step_dict = neuralnet.step(x=x_tr, y=y_tr, x_normals=x_normals, training=False)
        
        y_hat = step_dict['y_hat']
        residue = step_dict['residue']
        # for i, r in enumerate(residue):
          # print(residue[i].get_shape())
          # residue[i] = np.expand_dims(residue[i], -1)
          # residue[i] = (residue[i] - residue[i].min())/(residue[i].max() - residue[i].min())
          # residue[i] = cv2.normalize(residue[i], None, norm_type=cv2.NORM_L2)
        save_img(contents=[x_tr, y_hat, residue], \
            names=["Input\n(x)", "Restoration\n(x to x-hat)", "Residue Map"], \
            savename=os.path.join("training", "restoration", "%08d.png" %(epoch)))

        while(True):
            x_tr, y_tr, terminator, x_normals = dataset.next_train(batch_size)
            # before = process.memory_percent()
            step_dict = neuralnet.step(x=x_tr, y=y_tr, x_normals=x_normals, iteration=iteration, epoch=epoch, training=True)
            # after = process.memory_percent()
            if iteration % 1000 == 0:
              rec_1 = "Adversarial: %.3f | Reconstruction: %.3f | Total Variation: %.3f" % (step_dict['adv'], step_dict['r'], step_dict['tv'])
              rec_2 = "r1:%.3f | r2:%.3f | r3:%.3f" % (step_dict['r1'], step_dict['r2'], step_dict['r3'])
              rec = "%d iteration | G:%.3f, D:%.3f" \
                  %(iteration, step_dict['loss_g'], step_dict['loss_d'])
              with open("loss_record.txt", 'a') as f:
                f.write(f"{rec}\n{rec_1}\n{rec_2}\n")
            iteration += 1
            if(terminator): break
            # print("MEMORY CHANGE %.4f -> %.4f" % (before, after))
        rec = "Epoch [%d / %d] (%d iteration) | G:%.3f, D:%.3f" \
            %(epoch, epochs, iteration, step_dict['loss_g'], step_dict['loss_d'])
        print(rec)
        # r3 = step_dict['r3']
        # print(step_dict['l_a'])
        
        # with open("loss_record.txt", 'a') as f:
        #   f.write(f"{rec}\n")

        if epoch % 10 == 0:
          neuralnet.save_parameter(model='model_checker', epoch=epoch)
          # with open("model_record.txt", 'a') as f:
          #   f.write(f"model saved\n")
            
        
        

def test(neuralnet, dataset, batch_size):

    print("\nTest...")
    neuralnet.load_parameter(model='model_checker')

    make_dir(path="test")
    result_list = ["inbound", "outbound"]
    for result_name in result_list: make_dir(path=os.path.join("test", result_name))

    loss_list = []
    while(True):
        x_te, y_te, terminator = dataset.next_test(1)

        step_dict = neuralnet.step(x=x_te, y=y_te, training=False)
        y_hat, mse = step_dict['y_hat'], step_dict['mse']
        if(y_te[0] == 1):
            loss_list.append(mse)

        if(terminator): break

    loss_list = np.asarray(loss_list)
    loss_avg, loss_std = np.average(loss_list), np.std(loss_list)
    outbound = loss_avg + (loss_std * 3)
    print("Loss  avg: %.5f, std: %.5f" %(loss_avg, loss_std))
    print("Outlier boundary: %.5f" %(outbound))

    fcsv = open("test-summary.csv", "w")
    fcsv.write("class, loss, outlier\n")
    testnum = 0
    z_enc_tot, y_te_tot = None, None
    loss4box = [[], [], [], [], [], [], [], [], [], []]
    while(True):
        x_te, y_te, terminator = dataset.next_test(1)

        step_dict = neuralnet.step(x=x_te, y=y_te, training=False)
        y_hat, mse = step_dict['y_hat'], step_dict['mse']
        # y_hat = 
        # print(mse)
        # print(y_te[0, 0])
        loss4box[int(y_te[0, 0])].append(mse)

        outcheck = mse > outbound
        fcsv.write("%d, %.5f, %r\n" %(y_te, mse, outcheck))

        [h, w, c] = y_hat[0].shape
        # print(y_hat[0].shape)
        # print(x_te[0])
        canvas = np.ones((h, w*3, c), np.float32)
        canvas[:, :w, :] = x_te[0]
        canvas[:, w:w*2, :] = y_hat[0]
        canvas[:, w*2:, :] = (x_te[0]-y_hat[0])**2
        # print(canvas)
        if(outcheck):
            plt.imsave(os.path.join("test", "outbound", "%08d-%08d.png" %(testnum, int(mse))), gray2rgb(gray=canvas))
        else:
            plt.imsave(os.path.join("test", "inbound", "%08d-%08d.png" %(testnum, int(mse))), gray2rgb(gray=canvas))

        testnum += 1

        if(terminator): break

    # boxplot(contents=loss4box, savename="test-box.png")
