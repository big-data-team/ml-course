# About
Practical Machine Learning Course
* https://bigdatateam.org/ru/machine-learning-course

# Environment Configuration

1. Download [requirements.txt](requirements.txt)
2. Create environment:
```bash
export env_name="bdt-ml-course"
conda create -n $env_name python=3.10 -y
conda activate $env_name
# conda retires packages quite aggressively
# therefore pip is more robust way to setup environent:
pip install -r requirements.txt
# but if you love and trust conda, then call:
# conda install --file requirements.txt
```

For more information about Python virtual environments see:
* https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/
* https://docs.python.org/3/library/site.html
* https://docs.python.org/3/library/venv.html

See available conda environments with the help of:
```bash
conda info --envs
```

If you need to remove environment use the following command:
```bash
conda remove --name $env_name --all
```
