MobileNetV3(
  (features): Sequential(
    (0): Sequential(
      (0): Conv2d(3, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (1): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): h_swish(
        (sigmoid): h_sigmoid(
          (relu): ReLU6(inplace=True)
        )
      )
    )
    (1): InvertedResidual(
      (conv): Sequential(
        (0): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=16, bias=False)
        (1): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU(inplace=True)
        (3): Identity()
        (4): Conv2d(16, 16, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (5): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (2): InvertedResidual(
      (conv): Sequential(
        (0): Conv2d(16, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU(inplace=True)
        (3): Conv2d(64, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=64, bias=False)
        (4): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (5): Identity()
        (6): ReLU(inplace=True)
        (7): Conv2d(64, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (8): BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (3): InvertedResidual(
      (conv): Sequential(
        (0): Conv2d(24, 72, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): BatchNorm2d(72, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU(inplace=True)
        (3): Conv2d(72, 72, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=72, bias=False)
        (4): BatchNorm2d(72, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (5): Identity()
        (6): ReLU(inplace=True)
        (7): Conv2d(72, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (8): BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (4): InvertedResidual(
      (conv): Sequential(
        (0): Conv2d(24, 72, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): BatchNorm2d(72, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU(inplace=True)
        (3): Conv2d(72, 72, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), groups=72, bias=False)
        (4): BatchNorm2d(72, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (5): SELayer(
          (avg_pool): AdaptiveAvgPool2d(output_size=1)
          (fc): Sequential(
            (0): Linear(in_features=72, out_features=24, bias=True)
            (1): ReLU(inplace=True)
            (2): Linear(in_features=24, out_features=72, bias=True)
            (3): h_sigmoid(
              (relu): ReLU6(inplace=True)
            )
          )
        )
        (6): ReLU(inplace=True)
        (7): Conv2d(72, 40, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (8): BatchNorm2d(40, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (5): InvertedResidual(
      (conv): Sequential(
        (0): Conv2d(40, 120, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): BatchNorm2d(120, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU(inplace=True)
        (3): Conv2d(120, 120, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=120, bias=False)
        (4): BatchNorm2d(120, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (5): SELayer(
          (avg_pool): AdaptiveAvgPool2d(output_size=1)
          (fc): Sequential(
            (0): Linear(in_features=120, out_features=32, bias=True)
            (1): ReLU(inplace=True)
            (2): Linear(in_features=32, out_features=120, bias=True)
            (3): h_sigmoid(
              (relu): ReLU6(inplace=True)
            )
          )
        )
        (6): ReLU(inplace=True)
        (7): Conv2d(120, 40, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (8): BatchNorm2d(40, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (6): InvertedResidual(
      (conv): Sequential(
        (0): Conv2d(40, 120, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): BatchNorm2d(120, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU(inplace=True)
        (3): Conv2d(120, 120, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=120, bias=False)
        (4): BatchNorm2d(120, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (5): SELayer(
          (avg_pool): AdaptiveAvgPool2d(output_size=1)
          (fc): Sequential(
            (0): Linear(in_features=120, out_features=32, bias=True)
            (1): ReLU(inplace=True)
            (2): Linear(in_features=32, out_features=120, bias=True)
            (3): h_sigmoid(
              (relu): ReLU6(inplace=True)
            )
          )
        )
        (6): ReLU(inplace=True)
        (7): Conv2d(120, 40, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (8): BatchNorm2d(40, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (7): InvertedResidual(
      (conv): Sequential(
        (0): Conv2d(40, 240, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): BatchNorm2d(240, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): h_swish(
          (sigmoid): h_sigmoid(
            (relu): ReLU6(inplace=True)
          )
        )
        (3): Conv2d(240, 240, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=240, bias=False)
        (4): BatchNorm2d(240, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (5): Identity()
        (6): h_swish(
          (sigmoid): h_sigmoid(
            (relu): ReLU6(inplace=True)
          )
        )
        (7): Conv2d(240, 80, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (8): BatchNorm2d(80, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (8): InvertedResidual(
      (conv): Sequential(
        (0): Conv2d(80, 200, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): BatchNorm2d(200, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): h_swish(
          (sigmoid): h_sigmoid(
            (relu): ReLU6(inplace=True)
          )
        )
        (3): Conv2d(200, 200, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=200, bias=False)
        (4): BatchNorm2d(200, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (5): Identity()
        (6): h_swish(
          (sigmoid): h_sigmoid(
            (relu): ReLU6(inplace=True)
          )
        )
        (7): Conv2d(200, 80, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (8): BatchNorm2d(80, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (9): InvertedResidual(
      (conv): Sequential(
        (0): Conv2d(80, 184, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): BatchNorm2d(184, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): h_swish(
          (sigmoid): h_sigmoid(
            (relu): ReLU6(inplace=True)
          )
        )
        (3): Conv2d(184, 184, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=184, bias=False)
        (4): BatchNorm2d(184, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (5): Identity()
        (6): h_swish(
          (sigmoid): h_sigmoid(
            (relu): ReLU6(inplace=True)
          )
        )
        (7): Conv2d(184, 80, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (8): BatchNorm2d(80, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (10): InvertedResidual(
      (conv): Sequential(
        (0): Conv2d(80, 184, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): BatchNorm2d(184, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): h_swish(
          (sigmoid): h_sigmoid(
            (relu): ReLU6(inplace=True)
          )
        )
        (3): Conv2d(184, 184, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=184, bias=False)
        (4): BatchNorm2d(184, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (5): Identity()
        (6): h_swish(
          (sigmoid): h_sigmoid(
            (relu): ReLU6(inplace=True)
          )
        )
        (7): Conv2d(184, 80, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (8): BatchNorm2d(80, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (11): InvertedResidual(
      (conv): Sequential(
        (0): Conv2d(80, 480, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): BatchNorm2d(480, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): h_swish(
          (sigmoid): h_sigmoid(
            (relu): ReLU6(inplace=True)
          )
        )
        (3): Conv2d(480, 480, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=480, bias=False)
        (4): BatchNorm2d(480, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (5): SELayer(
          (avg_pool): AdaptiveAvgPool2d(output_size=1)
          (fc): Sequential(
            (0): Linear(in_features=480, out_features=120, bias=True)
            (1): ReLU(inplace=True)
            (2): Linear(in_features=120, out_features=480, bias=True)
            (3): h_sigmoid(
              (relu): ReLU6(inplace=True)
            )
          )
        )
        (6): h_swish(
          (sigmoid): h_sigmoid(
            (relu): ReLU6(inplace=True)
          )
        )
        (7): Conv2d(480, 112, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (8): BatchNorm2d(112, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (12): InvertedResidual(
      (conv): Sequential(
        (0): Conv2d(112, 672, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): BatchNorm2d(672, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): h_swish(
          (sigmoid): h_sigmoid(
            (relu): ReLU6(inplace=True)
          )
        )
        (3): Conv2d(672, 672, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=672, bias=False)
        (4): BatchNorm2d(672, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (5): SELayer(
          (avg_pool): AdaptiveAvgPool2d(output_size=1)
          (fc): Sequential(
            (0): Linear(in_features=672, out_features=168, bias=True)
            (1): ReLU(inplace=True)
            (2): Linear(in_features=168, out_features=672, bias=True)
            (3): h_sigmoid(
              (relu): ReLU6(inplace=True)
            )
          )
        )
        (6): h_swish(
          (sigmoid): h_sigmoid(
            (relu): ReLU6(inplace=True)
          )
        )
        (7): Conv2d(672, 112, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (8): BatchNorm2d(112, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (13): InvertedResidual(
      (conv): Sequential(
        (0): Conv2d(112, 672, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): BatchNorm2d(672, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): h_swish(
          (sigmoid): h_sigmoid(
            (relu): ReLU6(inplace=True)
          )
        )
        (3): Conv2d(672, 672, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), groups=672, bias=False)
        (4): BatchNorm2d(672, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (5): SELayer(
          (avg_pool): AdaptiveAvgPool2d(output_size=1)
          (fc): Sequential(
            (0): Linear(in_features=672, out_features=168, bias=True)
            (1): ReLU(inplace=True)
            (2): Linear(in_features=168, out_features=672, bias=True)
            (3): h_sigmoid(
              (relu): ReLU6(inplace=True)
            )
          )
        )
        (6): h_swish(
          (sigmoid): h_sigmoid(
            (relu): ReLU6(inplace=True)
          )
        )
        (7): Conv2d(672, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (8): BatchNorm2d(160, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (14): InvertedResidual(
      (conv): Sequential(
        (0): Conv2d(160, 960, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): BatchNorm2d(960, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): h_swish(
          (sigmoid): h_sigmoid(
            (relu): ReLU6(inplace=True)
          )
        )
        (3): Conv2d(960, 960, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=960, bias=False)
        (4): BatchNorm2d(960, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (5): SELayer(
          (avg_pool): AdaptiveAvgPool2d(output_size=1)
          (fc): Sequential(
            (0): Linear(in_features=960, out_features=240, bias=True)
            (1): ReLU(inplace=True)
            (2): Linear(in_features=240, out_features=960, bias=True)
            (3): h_sigmoid(
              (relu): ReLU6(inplace=True)
            )
          )
        )
        (6): h_swish(
          (sigmoid): h_sigmoid(
            (relu): ReLU6(inplace=True)
          )
        )
        (7): Conv2d(960, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (8): BatchNorm2d(160, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (15): InvertedResidual(
      (conv): Sequential(
        (0): Conv2d(160, 960, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): BatchNorm2d(960, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): h_swish(
          (sigmoid): h_sigmoid(
            (relu): ReLU6(inplace=True)
          )
        )
        (3): Conv2d(960, 960, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=960, bias=False)
        (4): BatchNorm2d(960, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (5): SELayer(
          (avg_pool): AdaptiveAvgPool2d(output_size=1)
          (fc): Sequential(
            (0): Linear(in_features=960, out_features=240, bias=True)
            (1): ReLU(inplace=True)
            (2): Linear(in_features=240, out_features=960, bias=True)
            (3): h_sigmoid(
              (relu): ReLU6(inplace=True)
            )
          )
        )
        (6): h_swish(
          (sigmoid): h_sigmoid(
            (relu): ReLU6(inplace=True)
          )
        )
        (7): Conv2d(960, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (8): BatchNorm2d(160, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
  )
  (conv): Sequential(
    (0): Conv2d(160, 960, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (1): BatchNorm2d(960, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): h_swish(
      (sigmoid): h_sigmoid(
        (relu): ReLU6(inplace=True)
      )
    )
  )
  (avgpool): AdaptiveAvgPool2d(output_size=(1, 1))
  (classifier): Sequential(
    (0): Linear(in_features=960, out_features=1280, bias=True)
    (1): h_swish(
      (sigmoid): h_sigmoid(
        (relu): ReLU6(inplace=True)
      )
    )
    (2): Dropout(p=0.2, inplace=False)
    (3): Linear(in_features=1280, out_features=1000, bias=True)
  )
)