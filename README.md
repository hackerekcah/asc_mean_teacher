# asc_mean_teacher
Acoustic Scene Classification with Device Mismatch, by Mean Teacher Approach

## 1. install
* create env and activate
```bash
conda create -n asc_mt python=3.6 pip
source activate asc_mt
```
* install pytorch torchvision from pytorch channel
```bash
conda install pytorch torchvision -c pytorch
```
* install requirements with pip
```bash
pip install -r requirements.txt
```
* if want to run jupyter notebook examples, install a `kernelspec` for env
conda install jupyter ipykernel
python -m ipykernel install --user --name asc_mt --display-name 'python3.6(asc_mt)'
## 2. data_manager
*NOTE: before use, config __data_manager.cfg__ properly*
### 2.1 extract and store feature in .h5 file
```
# generate .h5 files under data_manager/data_h5 
python data_manager/dcase18_taskb.py
# generate scaler .h5 under data_manager/data_h5
python data_manager/taskb_standrizer.py
```

### 2.2 load numpy data and labels using `data_manager`
* import and instantiate
```python
from data_manager.dcase18_taskb import Dcase18TaskbData
from data_manager.taskb_standrizer import TaskbStandarizer

data_manager = Dcase18TaskbData()
data_standarizer = TaskbStandarizer(data_manager=data_manager)
```
* load unormalized data and label using data_manager
```python
data_manager.load_dev(mode='train', devices='ab')
```
* load standarized data and label using `data_standrizer`
```python
data_standarizer.load_dev_standrized(mode='train', device='a', norm_device='a')
```


