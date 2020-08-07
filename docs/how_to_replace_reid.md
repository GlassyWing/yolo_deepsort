# How to replace Re-ID model

The repository use [fast-reid](https://github.com/JDAI-CV/fast-reid) as framework to support that.
So you need custom and train your model with [fast-reid](https://github.com/JDAI-CV/fast-reid) first.

Once you finish training your model with `fast-reid`,
then you can replace your reid model as you do in fast-reid:

1. Check out the branch `fast-reid`

2. Put your custom re-id model defining in `deep_sort/deep/modeling/backbones`, 
such as `resnet.py`, note that you must import your factory function in
`deep_sort/deep/modeling/backbones/__init__.py`,
E.g: `from .resnet import build_resnet_backbone`.
see more [fast-reid](https://github.com/JDAI-CV/fast-reid).

3. Put the config file generate by fast-reid in `config/deep`

4. Config the DeepSort

    There are serveral params that have to be setting: `config_file`, `MODEL.WEIGHTS`, `MODEL.DEVICE`, the config is need for `fast-reid`.

    ```python

    config = {"config_file": "config/deep/custom.yml",
              "opts": ["MODEL.WEIGHTS",
                       "weights/model_final.pth",
                       "MODEL.DEVICE",
                       "cuda:0"]}
    cfg = setup_cfg(config)

    tracker = DeepSort(cfg,
                       min_confidence=1,
                       nn_budget=30,
                       n_init=3,
                       max_iou_distance=0.7,
                       max_dist=0.5,
                       max_age=30)
    ```

5. Finished
   
   Now you can run the tracker with your Re-ID model.
