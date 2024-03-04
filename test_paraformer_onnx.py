from funasr_onnx import Paraformer
import time

model_dir = "./paraformer-large"
#model_dir = "./speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online-onnx"
model_dir = "./speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch" #Paraformer-large-pytorch
#model_dir = "./speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
#model_dir = "./speech_paraformer-large_asr_nat-zh-cantonese-en-16k-vocab8501-online"
print("model: ", model_dir)
model = Paraformer(model_dir, batch_size=1, quantize=True)

wav_path = ['./paraformer-large/asr_example.wav']
wav_path = ['./chatglm2.wav', 'chengyu.wav', 'chinese_10s_16k.wav', 'hongqiao.wav', 'Sheldon_10s.wav','jfk.wav']#, './zh_en.wav']


from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
inference_pipline2 = pipeline(
            task=Tasks.punctuation,
            model='damo/punc_ct-transformer_cn-en-common-vocab471067-large',
            model_revision="v1.0.0")
            #model='damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch',
            #model_revision="v1.1.7")
#result = model(wav_path)
#print(result)
for wavfile in wav_path:
    st = time.time()
    result = model(wavfile)
    end = time.time()
    print(f'Inference time: {end-st} s')
    print(result)
    print(result[0]['preds'][0])

    st2 = time.time()
    rec_result = inference_pipline2(text_in=result[0]['preds'][0])
    end2 = time.time()
    print(f'Inference time: {end2-st2} s')
    print(rec_result)
    print(f'Total Inference time: {end-st+end2-st2} s')
    print("-----------------------\n")
