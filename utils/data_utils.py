from jittor.dataset import Dataset
import jittor.transform as transforms
from PIL import Image

#
def get_dataset(dataroot):
 
    print("Warning: Jittor version requires pre-downloaded CIFAR10 dataset")
    return None, None


class Custom_dataset(Dataset):

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.ImageNormalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    def __init__(self, data, targets, transform=transform_test):
     
        super().__init__()
        self.data = data
        self.targets = targets
        self.n = len(list(targets))
        self.index = list(range(self.n))
        self.transform = transform
   
        self.batch_size = 1
        self.shuffle = False
        self.num_workers = 0
        # 计算batch_len
        self.batch_len = (self.n + self.batch_size - 1) // self.batch_size
 
        class ValueObj:
            def __init__(self, value=0):
                self.value = value
        self.gid_obj = ValueObj(0)
     
        self.num_idle = ValueObj(0)
  
        class Gid:
            def get_lock(self):
         
                class Lock:
                    def __enter__(self):
                        pass
                    def __exit__(self, *args):
                        pass
                return Lock()
            def get_obj(self):
 
                class Obj:
                    def __init__(self):
                        self.value = 0
                return Obj()
        self.gid = Gid()
   
        class IdQueue:
            def stop(self):
                pass
        self.idqueue = IdQueue()
     
        class Condition:
            def wait(self, timeout=None):
                pass
            def notify(self):
                pass
        self.num_idle_c = Condition()
        # 初始化其他可能需要的属性
        self.workers = []
        self.total_len = self.n
        self.drop_last = False
        self.started = False
        self.worker_count = 0
         
        self.data_queue = None
        self.result_queue = None
        self.stop_sign = False
        self.batch_count = 0
        self.sample_count = 0 
        self.lock = None
        self.batch_indices = []
        self.current_batch = 0

    def __getitem__(self, i):
        img = self.data[i]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)

        return img, self.targets[i], self.index[i]

    def __len__(self):
        return self.n

    def update_label(self, noise_label):
        self.targets[:] = noise_label[:]
