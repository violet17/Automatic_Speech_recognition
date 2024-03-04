# Automatic_Speech_recognition
FunASR--Best ASR in 1H'2024


pip install funasr-onnx  
pip install kaldi_native_fbank


ASR 载入加速，不做以下修改载入模型需要40s以上， 修改完载入模型大约需要6s：
（1）python 环境包中 funasr_onnx utils.py 中的 
347行  read_yaml 改动（可以减少十秒左右），使用CSafeLoader：
data = yaml.load(f, Loader=yaml.Loader) =》data = yaml.load(f, Loader=yaml.CSafeLoader)

301行  code_mix_split_words_jieba 中  
jieba.load_userdict(seg_dict_file) 改成 jieba.load_userdict_asr(seg_dict_file)
（2）python 环境包中 jieba 包中  __init__.py 中 400多行 添加：（可以减少十几秒）
    def load_userdict_asr(self, f):
        import json
        with open(f, 'r', encoding="utf8") as jsonf:
            self.FREQ = json.load(jsonf)

        self.total = 120424018
        self.initialized = True
        
        print("111111111111 self.dictionary:", self.dictionary,  len(self.FREQ), self.total, self.user_word_tag_tab, self.initialized)
500多行添加：load_userdict_asr = dt.load_userdict_asr
（3） CT-transformer 模型中使用新版的 jieba_usr_dict  大约46.5MB
