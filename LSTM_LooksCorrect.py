# LSTM Implementation Review

def lstm_cell(x, prev_h, prev_c, Wx, Wh, b):
    a = x @ Wx + prev_h @ Wh + b  # Use 'x', 'Wx', and 'prev_h'
    i, f, o, g = np.split(a, 4, axis=1) 
    i, f, o, g = sigmoid(i), sigmoid(f), sigmoid(o), np.tanh(g)
    next_c = f * prev_c + i * g
    next_h = o * np.tanh(next_c)
    cache = x, prev_h, prev_c, Wx, Wh, b, a, i, f, o, g, next_c
    return next_h, next_c, cache

def lstm(x, prev_h, prev_c, Wx, Wh, b):
    cache = []
    h = []
    for i in range(x.shape[0]):
        next_h, next_c, next_cache = lstm_cell(x[i:i+1], prev_h, prev_c, Wx, Wh, b)
        prev_h, prev_c = next_h, next_c
        cache.append(next_cache)
        h.append(next_h)
    h = np.concatenate(h, axis=0)
    return h, cache

def lstm_step_backward(dnext_h, dnext_c, cache):
    x, prev_h, prev_c, Wx, Wh, b, a, i, f, o, g, next_c = cache
    d1 = o * (1 - np.tanh(next_c) ** 2) * dnext_h + dnext_c
    dprev_c = f * d1
    dop = np.tanh(next_c) * dnext_h
    dfp = prev_c * d1
    dip = g * d1
    dgp = i * d1
    do = o * (1 - o) * dop
    df = f * (1 - f) * dfp
    di = i * (1 - i) * dip
    dg = (1 - g ** 2) * dgp
    da = np.concatenate((di, df, dg, do), axis=1)
    db = np.sum(da, axis=0)
    dx = da.dot(Wx.T)
    dprev_h = da.dot(Wh.T)
    dWx = x.T.dot(da)
    dWh = prev_h.T.dot(da)
    return dx, dprev_h, dprev_c, dWx, dWh, db

def lstm_backward(dh, cache):
    N, H = dh.shape
    D = cache[0][0].shape[1]  # Input dimension
    dx = np.zeros((N, D))
    dh0 = np.zeros((1, H))
    dWx = np.zeros_like(cache[0][3])
    dWh = np.zeros_like(cache[0][4])
    db = np.zeros_like(cache[0][5])
    
    dh_prev = np.zeros((1, H))
    dc_prev = np.zeros((1, H))
    
    for i in reversed(range(N)):
        dx_step, dh_prev, dc_prev, dWx_step, dWh_step, db_step = lstm_step_backward(
            dh[i:i+1] + dh_prev, dc_prev, cache[i])
        dx[i] = dx_step
        dWx += dWx_step
        dWh += dWh_step
        db += db_step
    
    dh0 = dh_prev
    return dx, dh0, dWx, dWh, db

