# Taiwanese-speech-recognition-using-ssl-pre-trained-models

# 學號: 311511057 姓名: 張詔揚

# 使用Wav2vec & transformer來做台語語音辨認

本次的作業是上一次作業的延伸，由於測試資料會將上雜訊，因此若是只使用乾淨的訓練音檔來訓練模型，則會導致訓練好的模型無法辨認出加上雜訊的測試音檔，
因此這次作業會採用wav2vec的預訓練模型來幫助我們去求取音檔中的特徵，除此之外，還會針對一部分的訓練音檔進行加雜訊的資料擴增，如此一來就能夠確保
在finetune後，模型不會只能辨識出乾淨的音檔，加雜訊的音檔反而辨識不出來的問題。

## 資料規格:

1. 單人女聲聲音（高雄腔）
2. 輸入：台語語音音檔（格式：wav檔, 22 kHz, mono, 32 bits） 
3. 輸出：台羅拼音（依教育部標準）

## 環境設置:
與先前的espnet的環境設置大致相同，不過因為會使用到S3PRL以及fairseq等套件來求音檔參數，因此必須安裝相關套件，指令如下:

```sh
$ Install S3PRL by install_s3prl.sh
$ Install fairseq by install_fairseq.sh
```

## Data-Preprocessing

The steps for data preprocessing:

1. 由於音檔格式為（wav檔, 22 kHz, mono, 32 bits），因此使用 sox 將音檔轉成（wav檔, 16 kHz, mono, 16 bits）
2. 使用kaggle上提供的程式，將訓練音檔加上雜訊，使訓練資料能夠擴增，幫助模型辨認雜訊音檔

## 事前準備:

1. 在run.sh中添加參數--feats_normalize uttmvn減少collect_stats花費時間
2. 確認各種驅動程式，以及套件的版本與ESPnet是相容的 (ex: pytorch、cuda...)
3. 修改/新增configuration file，可以從huggingface上選擇自己想要的wav2vec model，使用方式如下

```sh
freeze_param: ["frontend.upstream"]

frontend: s3prl
frontend_conf:
    frontend_conf:
        upstream: wav2vec2_url
        path_or_url: https://huggingface.co/s3prl/converted_ckpts/resolve/main/wav2vec2_base_s2st_en_librilight.pt
        # 從huggingface上下載pretrained model
    download_dir: ./wav2vec
    multilayer_feature: True

preencoder: linear
preencoder_conf:
    input_size: 768  # Note: If the upstream is changed, please change this value accordingly.
    output_size: 80
```

4. 在data.sh中，需要將下方程式碼改寫，改寫的目的是為了避免在輸出答案時，將空白消除，導致無法辨認
```sh
for x in train dev test; do
  cp data/${x}/text data/${x}/text.org
  #paste -d " " <(cut -f 1 -d" " data/${x}/text.org) <(cut -f 2- -d" " data/${x}/text.org | tr -d " ") \
  #    > data/${x}/text
  paste  data/${x}/text
  rm data/${x}/text.org
done
```

## 做完上述調整後，完整的Configure如下:
```sh
# Trained with Ampere A6000(48GB) x 2 GPUs. It takes about 10 days.
batch_type: numel
batch_bins: 3000000 #若GPU的空間不足，可以嘗試將此參數降低
accum_grad: 3
max_epoch: 80
patience: none
init: none
best_model_criterion:
-   - valid
    - acc
    - max
keep_nbest_models: 10
unused_parameters: true
freeze_param: [
"frontend.upstream"
]

frontend: s3prl
frontend_conf:
    frontend_conf:
        #upstream: wavlm_url  
        #path_or_url: https://huggingface.co/s3prl/converted_ckpts/resolve/main/wavlm_base_plus.pt
    #download_dir: ./wavlm
        upstream: wav2vec2_url
        path_or_url: https://huggingface.co/s3prl/converted_ckpts/resolve/main/wav2vec2_base_s2st_en_librilight.pt
    download_dir: ./wav2vec2
    multilayer_feature: True

preencoder: linear
preencoder_conf:
    input_size: 768  # Note: If the upstream is changed, please change this value accordingly.
    output_size: 80

encoder: conformer
encoder_conf:
    output_size: 512
    attention_heads: 8
    linear_units: 2048
    num_blocks: 12
    dropout_rate: 0.1
    positional_dropout_rate: 0.1
    attention_dropout_rate: 0.1
    input_layer: conv2d2
    normalize_before: true
    macaron_style: true
    pos_enc_layer_type: "rel_pos"
    selfattention_layer_type: "rel_selfattn"
    activation_type: "swish"
    use_cnn_module:  true
    cnn_module_kernel: 31

decoder: transformer
decoder_conf:
    attention_heads: 8
    linear_units: 2048
    num_blocks: 6
    dropout_rate: 0.1
    positional_dropout_rate: 0.1
    self_attention_dropout_rate: 0.1
    src_attention_dropout_rate: 0.1

model_conf:
    ctc_weight: 0.3
    lsm_weight: 0.1
    length_normalized_loss: false
    extract_feats_in_collect_stats: false   # Note: "False" means during collect stats (stage 10), generating dummy stats files rather than extract_feats by forward frontend.

optim: adam
optim_conf:
    lr: 0.00125
scheduler: warmuplr
scheduler_conf:
    warmup_steps: 40000


specaug: specaug
specaug_conf:
    apply_time_warp: true
    time_warp_window: 5
    time_warp_mode: bicubic
    apply_freq_mask: true
    freq_mask_width_range:
    - 0
    - 30
    num_freq_mask: 2
    apply_time_mask: true
    time_mask_width_range:
    - 0
    - 40
    num_time_mask: 2
```
如何進行訓練: 
```sh
./run.sh # 從頭跑一次訓練流程
./run.sh --stage n # 從上次中斷的地方執行， n = 11(從訓練階段開始跑)
```
## 訓練結果，以及各種方法的正確率比較:
![image](https://github.com/MachineLearningNTUT/taiwanese-speech-recognition-using-ssl-pre-trained-models-Hippo88902/blob/main/performance.jpg)
