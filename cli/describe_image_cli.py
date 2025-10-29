import argparse, os
from lib.describe_image import (
    describe_image
)

def main():
    parser = argparse.ArgumentParser(description="Image Description CLI")

    parser.add_argument("--image", type=str,  help="Path to an image file")
    parser.add_argument("--query", type=str, help="Text query to rewrite based on the image")

    args = parser.parse_args()

    if not os.path.exists(args.image):
        raise FileNotFoundError(f"Image file not found: {args.image}")

    text, tokens = describe_image(args.image, args.query)
    print(f"Rewritten query: {text}")
    print(f"Total tokens: {tokens}")

if __name__ == "__main__":
    main()