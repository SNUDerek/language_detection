import argparse

from language_detection import LanguageDetector


TEXT_SAMPLE = "안녕, 세상아! 다음은 한국어 텍스트 예시입니다."
DEFAULT_CKPT = "./experiments/wili2018/wili2018-checkpoint-000020.pt"

parser = argparse.ArgumentParser(description="train a language detection transformer classifier")
parser.add_argument("--checkpoint", type=str, default=DEFAULT_CKPT, help=f"path to checkpoint, default {DEFAULT_CKPT}")
parser.add_argument("--input", type=str, default=TEXT_SAMPLE, help="input text, defaults to Korean test string.")
parser.add_argument("--live", action="store_true", help="run an interactive terminal session (ignores --input)")
args = parser.parse_args()

lang_detector = LanguageDetector(args.checkpoint)

if args.live:
    print("\nTransformer Language Classification Demo by Derek Homel")
    print("this model was trained on the WiLI-2018 dataset.")
    print("enter a 1~2 sentence string to detect its language.")
    print("enter 'q', 'quit' or 'exit' to exit.")
    query = TEXT_SAMPLE
    while True:
        query = input(f"\nquery: ")
        if query in ("q", "quit", "exit"):
            break
        prediction = lang_detector.predict(query)
        print(f"language: {prediction}")
    print("\nthanks, bye!")

else:
    query = args.input
    prediction = lang_detector.predict(query)
    print(prediction)
