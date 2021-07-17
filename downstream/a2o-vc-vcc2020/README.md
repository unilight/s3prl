# Voice Conversion based on the Any-to-one ASR+TTS Framework

Main development: [Wen-Chin Huang](https://github.com/unilight) @ Nagoya University (2021).  
If you have any questions, please open an issue, or contact through email: wen.chinhuang@g.sp.m.is.nagoya-u.ac.jp

## Implementation

We implement the top system in the voice conversion challenge 2018 (VCC2018), as described in the following paper:  
[Liu, L., Ling, Z., Jiang, Y., Zhou, M., Dai, L. (2018) WaveNet Vocoder with Limited Training Data for Voice Conversion. Proc. Interspeech 2018, 1983-1987, DOI: 10.21437/Interspeech.2018-1190.](https://www.isca-speech.org/archive/Interspeech_2018/pdfs/1190.pdf)  

We made several modifications.
1. **Input**: instead of using the bottleneck features (BNFs) of a pretrained ASR model, we use the various upstreams provided in S3PRL.
2. **Output**: instead of using acoustic features extracted using a high-quality vocoder, STRAIGHT, we use the log-melspectrograms.
3. **Data**: we benchmark on the [VCC2020](https://github.com/nii-yamagishilab/VCC2020-database) dataset. 
4. **Training strategy**: instead of pretraining on a multispeaker dataset first, we directly trained on the target speaker training set.
5. **Vocoder**: instead of using the WaveNet vocoder, we used the [Parallel WaveGAN](https://arxiv.org/abs/1910.11480) (PWG) based on the [open source project](https://github.com/kan-bayashi/ParallelWaveGAN) developed by [kan-bayashi](https://github.com/kan-bayashi).

## Dependencies:

- `parallel-wavegan`
- `fastdtw`
- `pyworld`
- `pysptk`
- `jiwer`
- `resemblyzer`

## Usage

### Preparation
```
# Download the VCC2020 dataset.
cd <root-to-s3prl>/downstream/a2o-vc-vcc2020
cd data
./data_download.sh vcc2020/
cd ../

# Download the pretrained PWGs.
./pwg_download.sh ./
```

### Training
```
cd <root-to-s3prl>
python run_downstream.py -m train -n a2o_vc_vcc2020_<trgspk>_<upstream> -u <upstream> -d a2o-vc-vcc2020 -o "config.trgspk='<trgspk>'"
```
Along training, the converted samples generated using the Griffin-Lim algorithm will be saved in `result/downstream/a2o_vc_vcc2020_<trgspk>_<upstream>/<step>/test/wav/`.

### Decoding using PWG & Objective evaluation
```
cd <root-to-s3prl>/downstream/a2o-vc-vcc2020
./decode.sh pwg_task<1,2>/ ../../result/downstream/a2o_vc_vcc2020_<trgspk>_<upstream>/<step> <trgspk>
```
The generated samples using PWG will be saved in `result/downstream/a2o_vc_vcc2020_<trgspk>_<upstream>/<step>/test/pwg_wav/`.  
The output of the evaluation should look like:
```
Mean MCD, f0RMSE, f0CORR, DDUR, CER: 7.79 39.02 0.422 0.356 7.0 15.4
```
