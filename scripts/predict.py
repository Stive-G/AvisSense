"""Predict sentiment from the command line with the shared inference service."""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.inference import SentimentAnalyzer
from src.utils import clean_text


def print_prediction(text: str, prediction: dict, verbose: bool = False) -> None:
    if verbose:
        debug = prediction["debug"]
        tokens = debug["tokens"]
        print(f"  Tokens ({len(tokens)}) : {tokens[:15]}{'...' if len(tokens) > 15 else ''}")
        print(f"  Logits bruts : {debug['logits']}")
        print(f"  Probabilites : {prediction['probabilities']}")

    print(f"\n  Avis      : {text[:100]}{'...' if len(text) > 100 else ''}")
    print(f"  Sentiment : {prediction['label'].upper()}")
    print(f"  Confiance : {prediction['confidence']:.2%}")


def predict_and_print(analyzer: SentimentAnalyzer, text: str, verbose: bool) -> None:
    prediction = analyzer.predict(text, include_debug=verbose)
    print_prediction(text, prediction, verbose)


def interactive_mode(analyzer: SentimentAnalyzer) -> None:
    print("\nMode interactif - tapez un avis puis Entree ('q' pour quitter)\n")
    while True:
        try:
            text = clean_text(input("Votre avis > "))
        except (EOFError, KeyboardInterrupt):
            break
        if text.lower() in {"q", "quit", "exit"}:
            break
        if not text:
            print("  (texte vide, reessayez)")
            continue
        predict_and_print(analyzer, text, verbose=True)
        print()
    print("Au revoir !")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyse de sentiment d'un avis en francais"
    )
    parser.add_argument(
        "text",
        nargs="?",
        default=None,
        help="Avis a analyser. Sans argument : mode interactif.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Affiche les tokens, logits et probabilites",
    )
    args = parser.parse_args()

    try:
        analyzer = SentimentAnalyzer().load()
    except OSError as error:
        sys.exit(
            f"Erreur : modele introuvable ou inaccessible ({error}).\n"
            "Lancez d'abord : python scripts/train.py"
        )

    if args.text is None:
        interactive_mode(analyzer)
        return

    text = clean_text(args.text)
    if not text:
        sys.exit("Erreur : le texte est vide.")
    predict_and_print(analyzer, text, args.verbose)


if __name__ == "__main__":
    main()
