# You like karpathy? You like geohot? You love tinyGPT! ❤️

![tinyGPT](hotzilla.avif)

tinyGPT is an attempt to port karpathy's minGPT to geohot's tinygrad. It serves a few purposes:
- demonstrate API compatibility and diff between PyTorch and Tinygrad
- Identify missing features/APIs in Tinygrad
- Benchmark and compare performance

### Library Installation and Test

If you want to `import tinygpt` into your project:

```
git clone https://github.com/ziliangpeng/tinyGPT.git
cd tinyGPT
pip install -e .
```

After that, you can run the demo project to see the result:

```
cd project/adder
python adder.py
```

tinygrad allows you to choose hardware via env vars: `CLANG=1`, `CUDA=1`, `METAL=1`.

And you can choose `DEBUG=` level for increasing amount of debug log. Refer to [tinygrad](https://github.com/tinygrad/tinygrad/blob/master/docs-legacy/env_vars.md) for more env vars control.


### License

MIT
