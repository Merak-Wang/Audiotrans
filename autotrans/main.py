import argparse
import logging
import os

import autotrans.utils as utils
from autotrans.type import WhisperModel,WhisperMode

from test.config import TestArgs

def main():
    parser = argparse.ArgumentParser(
        description="Transcribe audio to text",
        formatter_class = argparse.RawDescriptionHelpFormatter,
    )

    logging.basicConfig(
        format="[transcribe:%(filename)s:L%(lineno)d] %(levelname)-6s %(message)s"
    )
    logging.getLogger().setLevel(logging.INFO)

    parser.add_argument(
        "-i",
        "--inputs",
        type=str,
        nargs="+",
        help="Inputs filenames/folders"
    )

    parser.add_argument(
        "-t","--transcribe",
        help="Transcribe audio to text",
        action=argparse.BooleanOptionalAction
    )

    parser.add_argument(
        "-c", "--cut",
        help="Cut the audio to transcribe",
        action=argparse.BooleanOptionalAction
    )

    parser.add_argument(
        "-m","--to-md",
        help="Convert .srt to .md for easier editing",
        action=argparse.BooleanOptionalAction,
    )

    parser.add_argument(
        "--force",
        help="Force write even if files exist",
        action=argparse.BooleanOptionalAction,
    )

    parser.add_argument(
        "--prompt",
        type=str,
        default="",
        help="initial prompt feed into whisper"
    )

    parser.add_argument(
        "--lang",
        type=str,
        default="zh",
        choices=[
            "zh",
            "en",
            "Afrikaans",
            "Arabic",
            "Armenian",
            "Azerbaijani",
            "Belarusian",
            "Bosnian",
            "Bulgarian",
            "Catalan",
            "Croatian",
            "Czech",
            "Danish",
            "Dutch",
            "Estonian",
            "Finnish",
            "French",
            "Galician",
            "German",
            "Greek",
            "Hebrew",
            "Hindi",
            "Hungarian",
            "Icelandic",
            "Indonesian",
            "Italian",
            "Japanese",
            "Kannada",
            "Kazakh",
            "Korean",
            "Latvian",
            "Lithuanian",
            "Macedonian",
            "Malay",
            "Marathi",
            "Maori",
            "Nepali",
            "Norwegian",
            "Persian",
            "Polish",
            "Portuguese",
            "Romanian",
            "Russian",
            "Serbian",
            "Slovak",
            "Slovenian",
            "Spanish",
            "Swahili",
            "Swedish",
            "Tagalog",
            "Tamil",
            "Thai",
            "Turkish",
            "Ukrainian",
            "Urdu",
            "Vietnamese",
            "Welsh",
        ],
        help="The output language of transcription",
    )

    parser.add_argument(
        "--whisper-mode",
        type=str,
        default=WhisperMode.WHISPER.value,
        choices=WhisperMode.get_values(),
        help="Whisper inference mode: whisper: run whisper locally",
    )

    parser.add_argument(
        "--whisper-model",
        type=str,
        default=WhisperModel.MEDIUM.value,
        choices=WhisperModel.get_values(),
        help="The whisper model used to transcribe.",
    )

    parser.add_argument(
        "--bitrate",
        type=str,
        default="320k",
        help="The bitrate to export the cutted Audio, such as 320k, 128k, or 1.411m",
    )

    parser.add_argument(
        "--encoding",
        type=str,
        default="utf-8",
        help="Document encoding format"
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cpu", "cuda"],
        help="Force to CPU or GPU for transcribing. In default automatically use GPU if available.",
    )

    parser.add_argument(
        "--vad",
        choices=["1", "0", "auto"],
        default="auto",
        help = "If or not use VAD(voice activity detection)",
    )


    args = parser.parse_args()

    #args=TestArgs()
    #args.inputs = [r"D:\media\胡彦斌 - 笔墨登场.mp3"]

    if args.transcribe:
        from .transcribe import Transcribe

        Transcribe(args).run()
    elif args.to_md:
        from .utils import trans_srt_to_md

        if len(args.inputs) == 2:
            [input_1, input_2] = args.inputs
            base, ext = os.path.splitext(input_1)
            if ext != ".srt":
                input_1, input_2 = input_2, input_1
            trans_srt_to_md(args.encoding, args.force, input_1, input_2)
        elif len(args.inputs) == 1:
            trans_srt_to_md(args.encoding, args.force, args.inputs[0])
        else:
            logging.warning(
                "Wrong number of files, please pass in a .srt file or an additional video file"
            )
    else:
        logging.warning("No action, use -c, -t or -d")




if __name__ == "__main__":
    main()



















