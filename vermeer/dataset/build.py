from dataset.ca_image import build_ca_code


def build_dataset(args, **kwargs):
    # images
    if args.dataset == 'ca_code':
        return build_ca_code(args, **kwargs)
    
    raise ValueError(f'dataset {args.dataset} is not supported')