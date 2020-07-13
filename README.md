# SMT

English to Odia statistical machine translation

## Quick start
```python
git clone https://github.com/MTEnglish2Odia/SMT.git
cd SMT

# install required packages preferably in a virtual environment
pip install -r requirements.txt

# train SMT model
cd src
python train.py  # it creates "model.pkl" in "models" dir

# web app
cd ../
python controller.py  # open http://127.0.0.1:31137/translate in browser
```

## Snapshot of web app
<img src="/snapshot.png" width="50%" height="50%"/>


[LICENSE](https://github.com/MTEnglish2Odia/SMT/blob/master/LICENSE)
