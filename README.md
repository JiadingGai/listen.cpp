1. How a torch weight tensor is dumped to disk as a binary file.
gai0 = x.numpy().astype('single').flatten()
gai0.tofile("__gold_input.gaibin")
yyy = self.conv1(x)


2. in __init__.py

    with (
        io.BytesIO(checkpoint_file) if in_memory else open(checkpoint_file, "rb")
    ) as fp:
        checkpoint = torch.load(fp, map_location=device)
        for k,v in checkpoint['model_state_dict'].items():
            print(k, "=", v.shape)
            if k == "encoder.conv2.weight" or k == "encoder.conv2.bias":
                print(v)
                import pdb; pdb.set_trace()
                out_fn = k.replace('.', '_') + '.gaibin'
                v.numpy().astype('single').tofile(out_fn)

    del checkpoint_file

    dims = ModelDimensions(**checkpoint["dims"])
