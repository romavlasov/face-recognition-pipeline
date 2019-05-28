# Face Recognition Pipeline

1. Download and extract [dataset](https://www.dropbox.com/s/wpx6tqjf0y5mf6r/faces_ms1m-refine-v2_112x112.zip?dl=0) (MS1M refined by Insightface, more info [here](https://github.com/deepinsight/insightface/wiki/Dataset-Zoo))

2. Prepare dataset <code> python tools/process_records.py --folder "path to extracted dataset" </code>

3. Update/create configuration file. Set up path to extracted dataset in configuration file (folder: "extracted dataset")

4. Train model <code>python train.py --config "configuration file"</code>

5. Test model <code>python test.py --config "configuration file"</code>

Results on trained models:

| Models        | LFW      | CFP_FP   | AGEDB_30 |
| ------------- |:--------:|:--------:|---------:|
| [MobileNet](https://www.dropbox.com/s/pjgpefh05q3bdko/mobilenet_256d_30e.pth?dl=0)     | 0.9958   | 0.9573   | 0.9650   |
| [Resnet34](https://www.dropbox.com/s/pgbm5xrsmgtg8z0/resnet34_256d_30e.pth?dl=0)      | 0.9975   | 0.9764   | 0.9760   |
| [SEResNeXt50](https://www.dropbox.com/s/aq6imy2o74md6nx/se_resnext50_256d_30e.pth?dl=0)   | 0.9975   | 0.9877   | 0.9792   |
