import argparse
from lib.multimodal_search import verify_image_embedding_command, image_search_command
    

def main():
    parser = argparse.ArgumentParser(description="Image Description CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    verify_image_embed_parser = subparsers.add_parser("verify_image_embedding", help="Verifies or embeds image")
    verify_image_embed_parser.add_argument("image", type=str, help="Image path for embedding")

    image_search_parser = subparsers.add_parser("image_search", help="Search docs in database using image")
    image_search_parser.add_argument("image", type=str, help="Image path for search")

    args = parser.parse_args()

    match args.command:
        case "verify_image_embedding":
            verify_image_embedding_command(args.image)
        case "image_search":
            image_search_command(args.image)
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()