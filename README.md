
# Installation
```
mamba env crete -f mf_swarm_base.yml
conda run --live-stream -n mf_swarm_base pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
python src/base_benchmark.py ~/data/dimension_db 1 ~/data/mf_datasets 30 0.1
```

# Molecular Function Prediction Benchmarking

|      **Models**      | **Total Feature Length** | **Classifier Parameters (millions)** | **AUPRC Score (W)** | **ROC AUC Score (W)** | **F1 Score (W)** | **Input Model Parameters (millions)** |
|:--------------------:|:------------------------:|:-------------------------:|:-------------------:|:---------------------:|:----------------:|:--------------------------:|
|       TAXA 256       | 2048                     | 6.430744                  | 0.8006              | 0.9094                | 0.7507           | 3650.0                     |
|       TAXA 128       | 1920                     | 6.785335                  | 0.796               | 0.9051                | 0.7487           | 3650.0                     |
|  ANKH BASE+ESM2 T33  | 2048                     | 3.684033                  | 0.7802              | 0.8888                | 0.7283           | 1300.0                     |
| ANKH BASE+ANKH LARGE | 2304                     | 4.066294                  | 0.7796              | 0.8892                | 0.7317           | 2450.0                     |
|   ANKH BASE+PROTT5   | 1792                     | 5.885268                  | 0.778               | 0.8905                | 0.7289           | 3650.0                     |
|   ANKH LARGE+PROTT5  | 2560                     | 6.666396                  | 0.7767              | 0.8913                | 0.7307           | 4800.0                     |
|  ANKH LARGE+ESM2 T36 | 4096                     | 5.25067                   | 0.7765              | 0.8889                | 0.7252           | 4800.0                     |
|  ANKH BASE+ESM2 T36  | 3328                     | 5.451484                  | 0.776               | 0.8854                | 0.7266           | 3650.0                     |
|      ANKH LARGE      | 1536                     | 3.138024                  | 0.7753              | 0.8889                | 0.7272           | 1800.0                     |
|    ESM2 T36+PROTT5   | 3584                     | 3.970484                  | 0.7722              | 0.8884                | 0.7195           | 6000.0                     |
|       ANKH BASE      | 768                      | 2.067504                  | 0.7718              | 0.8831                | 0.7238           | 650.0                      |
|  ANKH LARGE+ESM2 T33 | 2816                     | 9.557021                  | 0.7714              | 0.8851                | 0.7249           | 2450.0                     |
|    ESM2 T33+PROTT5   | 2304                     | 4.159314                  | 0.7704              | 0.8883                | 0.72             | 3650.0                     |
|       ESM2 T36       | 2560                     | 6.902479                  | 0.7641              | 0.8824                | 0.7104           | 3000.0                     |
|   ESM2 T33+ESM2 T36  | 3840                     | 5.605464                  | 0.7636              | 0.8811                | 0.7143           | 3650.0                     |
|        PROTT5        | 1024                     | 2.268203                  | 0.7604              | 0.8842                | 0.7142           | 3000.0                     |
|       ESM2 T33       | 1280                     | 4.461325                  | 0.7583              | 0.8817                | 0.7075           | 650.0                      |
|       ESM2 T30       | 640                      | 1.137218                  | 0.7466              | 0.8765                | 0.6967           | 150.0                      |
|       ESM2 T12       | 480                      | 0.62872                   | 0.739               | 0.8707                | 0.6861           | 35.0                       |
|        ESM2 T6       | 320                      | 0.438478                  | 0.7162              | 0.8606                | 0.6638           | 8.0                        |

![ROC AUC x AUPRC](img/model_performance-AUPRC%20Score%20(W)_ROC%20AUC%20Score%20(W).png)
![Total Parameters x AUPRC](img/model_performance-Total%20Parameters_ROC%20AUC%20Score%20(W).png)
