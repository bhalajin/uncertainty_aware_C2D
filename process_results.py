with open('checkpoint/sym_0.2_weights_run2_cifar10_0.20_0.0_sym_acc.txt') as f:
    best_acc   = 0;
    best_epoch = -1;
    for line in f:
        str_test = line.split()
        epoch    = int(str_test[0].split(":")[1])
        accuracy = float(str_test[1].split(":")[1])
        if accuracy >= best_acc:
            best_acc   = accuracy
            best_epoch = epoch

print("best_epoch: %d" % best_epoch)
print("best_acc: %.2f" % best_acc)
