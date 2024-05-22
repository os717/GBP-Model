# --------------------------
# ------ Compute Unit ------
# --------------------------

mult_mtx_mtx = {
    '1': {
        'latency': 132,
        'lut': 2019,
        'ff': 1572,
        'bram': 0,
        'dsp': 6
    },
    '3': {
        'latency': 84,
        'lut': 2019,
        'ff': 1976,
        'bram': 0,
        'dsp': 18
    }
}

mult_mtx_vec = {
     '1': {
        'latency': 56,
        'lut': 1270,
        'ff': 1147,
        'bram': 0,
        'dsp': 6
    },
    '3': {
        'latency': 44,
        'lut': 1563,
        'ff': 1486,
        'bram': 0,
        'dsp': 12
    }
}

inv_mtx = {
    '1': {
        'latency': 204,
        'lut': 2360,
        'ff': 2785,
        'bram': 0,
        'dsp': 15
    },
    '3': {
        'latency': 143,
        'lut': 3371,
        'ff': 4229,
        'bram': 0,
        'dsp': 27
    }
}

arith_mtx = {
    '1': {
        'latency': 54,
        'lut': 1205,
        'ff': 495,
        'bram': 0,
        'dsp': 4
    },
    '3': {
        'latency': 39,
        'lut': 1614,
        'ff': 691,
        'bram': 0,
        'dsp': 6
    }
}

arith_vec = {
     '1': {
        'latency': 20,
        'lut': 387,
        'ff': 679,
        'bram': 0,
        'dsp': 2
    },
    '3': {
        'latency': 14,
        'lut': 614,
        'ff': 1359,
        'bram': 0,
        'dsp': 6
    }
}