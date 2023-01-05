import numpy as np
from einops import rearrange


def clear_pages(dirty_pages, autoencoder):
    dirty_pages = 1-dirty_pages
    chars = rearrange(dirty_pages, 'p (H h) (W w) -> (p H W) h w', W=80, H=114, h=32, w=32)
    whitespace_chars = np.average(chars, axis=(1, 2)) < 0.1
    encoded_chars = np.squeeze(autoencoder.encoder.predict(chars, batch_size=4096))
    cleared_chars = np.squeeze(autoencoder.decoder.predict(encoded_chars, batch_size=4096))
    cleared_chars[whitespace_chars] = 0
    result_pages = rearrange(cleared_chars, '(p H W) h w -> p (H h) (W w)', W=80, H=114, h=32, w=32)
    result_pages = (result_pages > 0.3) * 1.0
    result_pages = 1-result_pages
    return [page for page in result_pages]
