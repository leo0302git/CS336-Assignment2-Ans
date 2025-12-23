MODEL_SPECS = {
        "small":  dict(d_model=768,  d_ff=3072,  num_layers=12, num_heads=12),
        "medium": dict(d_model=1024, d_ff=4096,  num_layers=24, num_heads=16),
        "large":  dict(d_model=1280, d_ff=5120,  num_layers=36, num_heads=20),
        "xl":     dict(d_model=1600, d_ff=6400,  num_layers=48, num_heads=25),
        "2.7b":   dict(d_model=2560, d_ff=10240, num_layers=32, num_heads=32),
    }
for k,v in MODEL_SPECS.items():
    print(k)
    print(v)