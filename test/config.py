import logging
import os

# 定义一个日志收集器
logger = logging.getLogger()
# 设置收集器的级别，不设定的话，默认收集warning及以上级别的日志
logger.setLevel("DEBUG")
# 设置日志格式
fmt = logging.Formatter("%(filename)s-%(lineno)d-%(asctime)s-%(levelname)s-%(message)s")
# 设置日志处理器-输出到文件,并且设置编码格式
if not os.path.exists("./log"):
    os.makedirs("./log")
file_handler = logging.FileHandler("./log/log.txt", encoding="utf-8")
# 设置日志处理器级别
file_handler.setLevel("DEBUG")
# 处理器按指定格式输出日志
file_handler.setFormatter(fmt)
# 输出到控制台
ch = logging.StreamHandler()
# 设置日志处理器级别
ch.setLevel("DEBUG")
# 处理器按指定格式输出日志
ch.setFormatter(fmt)
# 收集器和处理器对接，指定输出渠道
# 日志输出到文件
logger.addHandler(file_handler)
# 日志输出到控制台
logger.addHandler(ch)

TEST_MEDIA_PATH = "./media/"
TEST_CONTENT_PATH = "./content/"
TEST_MEDIA_FILE = [
    "test001.mp3",
]

TEST_MEDIA_FILE_LANG = ["test001.mp3"]
TEST_MEDIA_FILE_SIMPLE = ["test001.mp3"]


class TestArgs:
    def __init__(self):
        self.inputs = []
        self.bitrate = "320k"
        self.encoding = "utf-8"
        self.sampling_rate = 16000
        self.lang = "zh"
        self.prompt = "lyric of music："
        self.whisper_model = "medium"
        self.device = "cuda"
        self.vad = False
        self.force = False
        self.whisper_mode = (
            "faster"
            #"whisper"
        )
        self.openai_rpm = 3