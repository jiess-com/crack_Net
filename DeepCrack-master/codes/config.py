from pprint import pprint
import os
import setproctitle

class Config:
    name = ''

    gpu_id = '3'

    setproctitle.setproctitle("%s" % name)

    # path
    # train_data_path = '/data/glh/total_data/crack500/train/train.txt'
    train_data_path='/data/glh/crack260/crack260.txt'
    # val_data_path = '/data/glh/total_data/crack500/val/val.txt'
    val_data_path='/data/glh/crack315/crack315.txt'
    checkpoint_path = '/data/glh/dcrack/crackls/'
    log_path = 'log'
    saver_path = os.path.join(checkpoint_path, name)
    max_save = 20

    # visdom
    vis_env = 'DCrack'
    port = 8097
    vis_train_loss_every = 40
    vis_train_acc_every = 40
    vis_train_img_every = 120
    val_every = 200

    # training
    epoch = 500
    # pretrained_model = '/data/glh/dcrack/crackls/checkpoints/_0000178.pth'
    # pretrained_model="/data/glh/dcrack/crackls/checkpoints/_0000018.pth"
    pretrained_model=''
    weight_decay = 0
    lr_decay = 0.1
    lr = 1e-3
    momentum = 0.9
    use_adam = True  # Use Adam optimizer
    train_batch_size = 2
    val_batch_size = 2
    test_batch_size = 1

    acc_sigmoid_th = 0.5
    pos_pixel_weight = 1

    # checkpointer
    save_format = ''
    save_acc = -1
    save_pos_acc = -1
    save_ods=0.4
    saveois=0.4
    def _parse(self, kwargs):
        state_dict = self._state_dict()
        for k, v in kwargs.items():
            if k not in state_dict:
                raise ValueError('UnKnown Option: "--%s"' % k)
            setattr(self, k, v)

        print('======user config========')
        pprint(self._state_dict())
        print('==========end============')

    def _state_dict(self):
        return {k: getattr(self, k) for k, _ in Config.__dict__.items() \
                if not k.startswith('_')}

    def show(self):
        print('======user config========')
        pprint(self._state_dict())
        print('==========end============')
