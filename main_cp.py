from ocr import ImageTextExtractor

def main():
    extractor = ImageTextExtractor(img)
    text = extractor.extract_text()

if __name__ == "__main__":
    main()