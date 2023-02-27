# generator, model_nlayer_nchannel_nfft

# generator
# model_nlayer_nchannel
UNet_6_128 = {
    # (in_channels, out_channels, kernel_size, stride, padding)
    "encoder":
        [[  0,  32, (5, 2), (2, 1), (1, 1)],
         ( 32,  64, (5, 2), (2, 1), (2, 1)),
         ( 64, 128, (5, 2), (2, 1), (2, 1))],
    # (in_channels, out_channels, kernel_size, stride, padding, output_padding, is_last)
    "decoder":
        [(256,  64, (5, 2), (2, 1), (2, 0), (1, 0)),
         (128,  32, (5, 2), (2, 1), (2, 0), (1, 0)),
         [ 64,   0, (5, 2), (2, 1), (1, 0), (0, 0), True]]
}

# discriminator
# model_nlayer_nchannel_nfft
CNN_6_128_512 = [
    # (in_channels, out_channels, kernel_size, stride, padding)
    [  2,   8, (5, 5), (2, 2), (0, 0)],
    (  8,  16, (5, 5), (2, 2), (0, 0)),
    ( 16,  32, (5, 5), (2, 2), (0, 0)),
    ( 32,  64, (5, 5), (2, 2), (0, 0)),
    ( 64, 128, (5, 5), (2, 2), (0, 0)),
    (128, 128, (5, 5), (2, 2), (0, 0))
]