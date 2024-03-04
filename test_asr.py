from funasr_onnx import Paraformer,CT_Transformer
import time

asr_model_path = "./paraformer-large"

print("model: ", asr_model_path)
asr_model = Paraformer(asr_model_path, batch_size=1, quantize=True)

ct_model_dir = "./punc_ct-transformer_cn-en-common-vocab471067-large-onnx"
print("model: ", ct_model_dir )
ct_model = CT_Transformer(ct_model_dir, batch_size=1,quantize=True)

print("load model ---------------------Done")
wav_path = ['./chatglm2.wav', 'chengyu.wav', 'chinese_10s_16k.wav', 'hongqiao.wav', 'Sheldon_10s.wav','jfk.wav']


for wavfile in wav_path:
    st = time.time()
    result = asr_model(wavfile)
    end = time.time()
    print(f'Inference time: {end-st} s')
    print(result)
    print(result[0]['preds'][0])
    
    st2 = time.time()
    rec_result = ct_model(text=result[0]['preds'][0])
    end2 = time.time()
    print(f'Inference time: {end2-st2} s')
    print(rec_result)
    print(rec_result[0])#这是ASR之后的结果
    print(f'Total Inference time: {end-st+end2-st2} s')
    print("-----------------------\n")
