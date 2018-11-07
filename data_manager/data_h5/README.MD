
* TaskbDev.h5
```python
    [mode][wav_name]
        .attrs['label']
        .attrs['device']
        .attrs['venue']
```
* TaskbDevMatrix.h5
```python
    [mode][device]['data']
    [mode][device]['label']
```
* TaskbDevMatrixFnames.h5
```python
    [mode][device]['data']
    [mode][device]['label']
    [mode][device]['fnames']
```
* TaskbDevMatrixScale.h5
```python
    [mode][device]['mu']
    [mode][device]['sigma']
```
* audio num for each device
```python
train a (6122, 40, 500)
train b (540, 40, 500)
train c (540, 40, 500)
train p (540, 40, 500)

test a (2518, 40, 500)
test b (180, 40, 500)
test c (180, 40, 500)
test p (180, 40, 500)
```


