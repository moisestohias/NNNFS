
def slow_shrink_bilinear(input_image: np.ndarray, out_h: int, out_w: int = None):
  """
  Shrink input image to the givin size
  :: args:
    + input_image: ndarray of shape (C,H,W) or (H,W,C)
  >Note: this only meant for demonstration pupouse,
  since it's really slow to perform this operation in pure Numpy,
  for practical purpouse you must you Pytorch or OpenCV which implmenet these operation in Cpp
  >Note: This operation can't be batched because the input images have different dimensions
  """
  if out_w is None: out_w = out_h

  if len(input_image.shape) == 3:
    if input_image.shape[2] == 3:
      HWC = True
      in_h, in_w, channels = input_image.shape
    elif input_image.shape[0] == 3:
      HWC = False
      input_image = input_image.transpose(1, 2, 0)
      in_h, in_w, channels = input_image.shape
    else: raise ValueError("Number of channels must be 3")

    h_scale = in_h / out_h
    w_scale = in_w / out_w

    output_image = np.zeros((out_h, out_w, channels), dtype=np.uint8)

    for i in range(out_h):
      for j in range(out_w):
        y = i * h_scale
        x = j * w_scale

        y_low, y_high = int(np.floor(y)), int(np.ceil(y))
        x_low, x_high = int(np.floor(x)), int(np.ceil(x))

        y_fraction = y - y_low
        x_fraction = x - x_low

        for c in range(channels):
          top_left = input_image[y_low, x_low, c]
          top_right = input_image[y_low, x_high, c]
          bottom_left = input_image[y_high, x_low, c]
          bottom_right = input_image[y_high, x_high, c]

          interpolated_value = (1 - y_fraction) * ((1 - x_fraction) * top_left + x_fraction * top_right) + \
          y_fraction * ((1 - x_fraction) * bottom_left + x_fraction * bottom_right)

          output_image[i, j, c] = interpolated_value

  if not HWC: output_image.transpose(2, 0, 1)
  return output_image

