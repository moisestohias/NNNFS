import os
from PIL import Image
from torchvision import transforms

from PIL import Image
import os
import numpy as np

class Dataset:
  def __init__(self, xs, ys, Shuffle:bool=False, OneHot:bool=False, classes:int=None):
    if Shuffle: xs, ys = shuffle_data(xs, ys)
    self.xs = xs
    self.ys = one_hot(ys, classes) if OneHot and not isinstance(classes, int) else ys 
    self.counter = 0
    self.size = xs.shape[0]
  def __len__(self): return self.size
  def __iter__(self): return self
  def __next__(self):
    yld = self.xs[self.counter], self.ys[self.counter]
    if self.counter < self.size-1: self.counter += 1
    else: raise StopIteration
    return yld
  def __getitem__(self,n): return list(zip(self.xs[n], self.ys[n]))
  def __repr__(self): return f"{self.__class__.__name__}(xs, ys)"

  # GPT-4
  @classmethod
  def ImageFolder(cls, path):
    xs = [] # Images
    ys = [] # Labels
    classes = {}
    class_idx = 0

    # Iterate through subdirectories
    for folder_name in os.listdir(path):
      folder_path = os.path.join(path, folder_name)
      if os.path.isdir(folder_path):
        # Assign class index
        classes[folder_name] = class_idx
        
        # Iterate through images in folder
        for file_name in os.listdir(folder_path):
          file_path = os.path.join(folder_path, file_name)
          if os.path.isfile(file_path):
            # Read image
            image = Image.open(file_path)
            image_array = np.array(image)
            xs.append(image_array)
            ys.append(class_idx)
        
        class_idx += 1

    return cls(np.array(xs), np.array(ys), classes=class_idx)


# ChatGPT
class Dataset:
  def __init__(self, xs, ys, Shuffle:bool=False, OneHot:bool=False, classes:int=None):
    if Shuffle: xs, ys = shuffle_data(xs, ys)
    self.xs = xs
    self.ys = one_hot(ys, classes) if OneHot and not isinstance(classes, int) else ys 
    self.counter = 0
    self.size = xs.shape[0]
  def __len__(self): return self.size
  def __iter__(self): return self
  def __next__(self):
    yld = self.xs[self.counter], self.ys[self.counter]
    if self.counter < self.size-1: self.counter += 1
    else: raise StopIteration
    return yld
  def __getitem__(self,n): return list(zip(self.xs[n], self.ys[n]))
  def __repr__(self): return f"{self.__class__.__name__}(xs, ys)"

  @classmethod
  def ImageFolder(cls, path, transform=None):
    xs = []
    ys = []
    classes = []

    for class_name in os.listdir(path):
      class_path = os.path.join(path, class_name)
      if os.path.isdir(class_path):
        classes.append(class_name)
        class_index = len(classes) - 1

        for image_name in os.listdir(class_path):
          image_path = os.path.join(class_path, image_name)
          if image_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            xs.append(image_path)
            ys.append(class_index)

    xs = sorted(xs)
    ys = sorted(ys)
      
    if transform is None:
      transform = transforms.Compose([
        transforms.Resize((224, 224)),
      ])
      
    transformed_xs = []
    for image_path in xs:
      image = Image.open(image_path)
      transformed_image = transform(image)
      transformed_xs.append(transformed_image)
    
    xs_tensor = np.stack(transformed_xs)
    ys_tensor = np.array(ys)
    
    return cls(xs_tensor, ys_tensor, classes=len(classes))

# Example usage
path_to_images = "/path/to/your/image/folder"
dataset = Dataset.ImageFolder(path_to_images)
