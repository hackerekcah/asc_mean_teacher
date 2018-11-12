# utils
## check_point.py
## history.py
## vis.py
* visualize trained **model filters**
* visualize **feature maps** given input and model architecture
### Plot filters
* only support 1st layer
```
vis.conv1_weight()
```

### Plot feature map
* chaining input, get feature map and show
```python
# plot batch=0,channel=1
vis.feed(x).fea_map('conv1').show(batch_idx=0, channel_idx=1)
# plot whole batch by setting `batch_idx=None`
vis.feed(x).fea_map('conv1').show(batch_idx=None, channel_idx=1)
```