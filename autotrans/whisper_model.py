import numpy as np
import srt
import datetime
import opencc

from tqdm import tqdm
from typing import Literal , Union , List
from abc import ABC,abstractmethod
from .type import SPEECH_ARRAY_INDEX,LANG

#将可能生成的繁体字转化成简体
cc = opencc.OpenCC("t2s")

class AbstractWhisperModel(ABC):
    def __init__(self,mode,sample_rate=16000):
        self.mode = mode
        self.sample_rate = sample_rate
        self.whisper_model = None

    @abstractmethod
    def load(self,*args,**kwargs):
        pass

    @abstractmethod
    def transcribe(self,*args,**kwargs):
        pass

    @abstractmethod
    def _transcribe(self,*args,**kwargs):
        pass

    @abstractmethod
    def gen_srt(self,*args,**kwargs):
        pass

class WhisperModel(AbstractWhisperModel):
    def __init__(self,sampling_rate=16000):
        super().__init__("whisper",sampling_rate)
        self.device = None

    def load(self,
             model_name: Literal["tiny", "base", "small", "medium", "large", "large-v2"] = "medium",
             device: Union[Literal["cpu","cuda"],None] = None,
             ):
        self.device = device

        import whisper

        self.whisper_model = whisper.load_model(model_name,device)

    def _transcribe(self,audio,seg,lang,prompt):
        r = self.whisper_model.transcribe(
            audio[int(seg["start"]):int(seg["end"])],
            task = "transcribe",
            language = lang,
            initial_prompt = prompt
        )
        r["origin_timestamp"]=seg
        return r

    def transcribe(self,
                    audio:np.ndarray,
                    speech_array_index: List[SPEECH_ARRAY_INDEX],
                    lang:LANG,
                    prompt:str
                    ):
        res=[]
        if self.device == "cpu" and len(speech_array_index) > 1:
            from multiprocessing import Pool

            #展示音频处理的进度条
            pbar = tqdm(total=len(speech_array_index))

            pool = Pool(processes=4)

            sub_res=[]
            for seg in speech_array_index:
                sub_res.append(
                    pool.apply_async(
                        self._transcribe(
                            audio,
                            seg,
                            lang,
                            prompt,
                        ),
                        callback=lambda x: pbar.update(),
                    )
                )
            pool.close()
            pool.join()
            pbar.close()
            res = [i.get() for i in sub_res]
        else:
            for seg in (speech_array_index
                        if len(speech_array_index)==1 else tqdm(speech_array_index)):
                r = self.whisper_model.transcribe(
                    audio[int(seg["start"]):int(seg["end"])],
                    task= "transcribe",
                    language=lang,
                    initial_prompt=prompt,
                    verbose=False if len(speech_array_index) == 1 else None
                )
                r["origin_timestamp"] = seg
                res.append(r)
        return res

    #transcribe_results:dict["text":decode(all_tokens),"segments":all_segments,"language":lang,"origin_timestamp":seg]
    def gen_srt(self,transcribe_results):
        subs=[]

        def _add_sub(start, end, text):
            subs.append(
                srt.Subtitle(
                    index=0,
                    start=datetime.timedelta(seconds=start),
                    end=datetime.timedelta(seconds=end),
                    content=cc.convert(text.strip()),
                )
            )

        prev_end = 0
        for r in transcribe_results:
            origin = r["origin_timestamp"]
            for s in r["segments"]:
                start = s["start"] + origin["start"] / self.sample_rate
                end = min(
                    s["end"] + origin["start"] / self.sample_rate,
                    origin["end"] / self.sample_rate,
                )
                if start > end:
                    continue
                # mark any empty segment that is not very short
                if start > prev_end + 1.0:
                    _add_sub(prev_end, start, "< No Speech >")
                _add_sub(start, end, s["text"])
                prev_end = end
        return subs

class FasterWhisperModel(AbstractWhisperModel):
    def __init__(self, sample_rate=16000):
        super().__init__("faster-whisper", sample_rate)
        self.device = None

    def load(
        self,
        model_name: Literal[
            "tiny", "base", "small", "medium", "large", "large-v2"
        ] = "large-v2",
        device: Union[Literal["cpu", "cuda"], None] = None,
    ):
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            raise Exception(
                "Please use faster mode(pip install '.[faster]') or all mode(pip install '.[all]')"
            )

        self.device = device if device else "cpu"
        self.whisper_model = WhisperModel(model_name, self.device)

    def _transcribe(self):
        raise Exception("Not implemented")

    def transcribe(
        self,
        audio: np.ndarray,
        speech_array_indices: List[SPEECH_ARRAY_INDEX],
        lang: LANG,
        prompt: str,
    ):
        res = []
        for seg in speech_array_indices:
            segments, info = self.whisper_model.transcribe(
                audio[int(seg["start"]) : int(seg["end"])],
                task="transcribe",
                language=lang,
                initial_prompt=prompt,
                vad_filter=False,
            )
            segments = list(segments)  # The transcription will actually run here.
            r = {"origin_timestamp": seg, "segments": segments, "info": info}
            res.append(r)
        return res

    def gen_srt(self, transcribe_results):
        subs = []

        def _add_sub(start, end, text):
            subs.append(
                srt.Subtitle(
                    index=0,
                    start=datetime.timedelta(seconds=start),
                    end=datetime.timedelta(seconds=end),
                    content=cc.convert(text.strip()),
                )
            )

        prev_end = 0
        for r in transcribe_results:
            origin = r["origin_timestamp"]
            for seg in r["segments"]:
                s = dict(start=seg.start, end=seg.end, text=seg.text)
                start = s["start"] + origin["start"] / self.sample_rate
                end = min(
                    s["end"] + origin["start"] / self.sample_rate,
                    origin["end"] / self.sample_rate,
                )
                if start > end:
                    continue
                # mark any empty segment that is not very short
                if start > prev_end + 1.0:
                    _add_sub(prev_end, start, "< No Speech >")
                _add_sub(start, end, s["text"])
                prev_end = end

        return subs











