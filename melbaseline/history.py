# encoding: utf-8
import numpy as np
import sys


def print_history(history):
    print('{:12} | {:12}'.format('bin_acc', 'val_bin_acc'))
    print('-------------+-------------')
    for i in range(len(history['binary_accuracy'])):
        print('{:0.10f} | {:0.10f}'.format(history['binary_accuracy'][i], history['val_binary_accuracy'][i]))
    print()
    print('{:12} | {:12}'.format('loss', 'val_loss'))
    print('-------------+-------------')
    for i in range(len(history['loss'])):
        print('{:0.10f} | {:0.10f}'.format(history['loss'][i], history['val_loss'][i]))
    print()
    sys.stdout.flush()
    print_simple_bar_graph('bin_acc', history['binary_accuracy'])
    print()
    print_simple_bar_graph('val_bin_acc', history['val_binary_accuracy'])
    print()
    print_simple_bar_graph('loss', history['loss'])
    print()
    print_simple_bar_graph('val_loss', history['val_loss'])
    sys.stdout.flush()


def print_simple_bar_graph(title, data):
    data = np.array(data)
    minimum = np.min(data)
    maximum = np.max(data)
    size = 50
    if maximum == minimum:
        data.fill(size)
    else:
        data = (data-minimum)/maximum * size

    # Eighths-blocks don't seem to be supported by Google ML Engine
    # block = '█' * size
    # eights = [' ', '▏', '▎', '▍', '▌', '▋', '▊', '▉']
    block = '#' * size
    eights = [' ', ' ', ' ', ' ', '#', '#', '#', '#']
    line = '-' * (size + 5)
    print(line)
    print(title)
    print(line)
    for i in range(data.shape[0]):
        d = data[i]
        l = int(d)
        f = int((d - int(d)) * 8)
        print('{:3} {}{}'.format(i+1, block[:l], eights[f]))
    print(line)
    print('Min={}, Max={}'.format(minimum, maximum))
    print(line)

