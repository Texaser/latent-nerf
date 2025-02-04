from loguru import logger


def get_encoder(encoding, input_dim=3,
                multires=6,
                degree=4,
                num_levels=16, level_dim=2, base_resolution=16, log2_hashmap_size=19, desired_resolution=2048,
                align_corners=False,
                iteration = 0,
                **kwargs):
    logger.info(f'Loading {encoding} encoding (compiling might take a while)...')

    if encoding == 'None':
        return lambda x, **kwargs: x, input_dim

    elif encoding == 'frequency':
        from .encoders.freqencoder import FreqEncoder
        encoder = FreqEncoder(input_dim=input_dim, degree=multires)

    elif encoding == 'sphere_harmonics':
        from .encoders.shencoder import SHEncoder
        encoder = SHEncoder(input_dim=input_dim, degree=degree)

    elif encoding == 'hashgrid':
        from .encoders.gridencoder import GridEncoder
        encoder = GridEncoder(input_dim=input_dim, num_levels=num_levels, level_dim=level_dim,
                              base_resolution=base_resolution, log2_hashmap_size=log2_hashmap_size,
                              desired_resolution=desired_resolution, gridtype='hash', align_corners=align_corners)

    elif encoding == 'tiledgrid':
        from .encoders.gridencoder import GridEncoder
        encoder = GridEncoder(input_dim=input_dim, num_levels=num_levels, level_dim=level_dim,
                              base_resolution=base_resolution, log2_hashmap_size=log2_hashmap_size,
                              desired_resolution=desired_resolution, gridtype='tiled', align_corners=align_corners)

    elif encoding == 'triplane':
        from .encoders.gridencoder import MiniTriplane, MultiScaleTriplane
        # encoder = MiniTriplane(input_dim=input_dim)
        encoder = MultiScaleTriplane(input_dim=input_dim)

    elif encoding == 'triplane_pooling':
        from .encoders.gridencoder import MultiScaleTriplane_Pooling
        encoder = MultiScaleTriplane_Pooling(input_dim=input_dim, iteration=iteration)

    
    else:
        raise NotImplementedError(
            'Unknown encoding mode, choose from [None, frequency, sphere_harmonics, hashgrid, tiledgrid]')
    logger.info('\tDone!')

    return encoder, encoder.output_dim
