# asc_mean_teacher
Acoustic Scene Classification with Device Mismatch, by Mean Teacher Approach

## install
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
## data_manager
*before use, config __data_manager.cfg__ properly*
```python
from data_manager.dcase18_taskb import Dcase18TaskbData
from data_manager.taskb_standrizer import TaskbStandarizer

data_manager = Dcase18TaskbData()
data_standarizer = TaskbStandarizer(data_manager=data_manager)

```
