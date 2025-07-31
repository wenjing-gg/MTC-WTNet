# PDF to Image Conversion Instructions

To display your PDF directly on the GitHub main page, you need to convert the PDF to an image format.

## Method 1: Online Conversion
1. Go to https://pdf2png.com/ or https://smallpdf.com/pdf-to-jpg
2. Upload your `main.pdf` file
3. Download the first page as PNG/JPG
4. Rename it to `paper_preview.png`
5. Replace the placeholder image in the `img/` folder

## Method 2: Using Python (if you have it installed)
```python
import fitz  # PyMuPDF
from PIL import Image

# Open PDF
pdf_document = fitz.open("main.pdf")
page = pdf_document[0]  # First page

# Convert to image
pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x zoom for better quality
img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
img.save("paper_preview.png")
pdf_document.close()
```

## Method 3: Using ImageMagick (command line)
```bash
convert main.pdf[0] -density 300 paper_preview.png
```

## Recommended Settings:
- **Format**: PNG (for better quality) or JPG
- **Resolution**: 300 DPI or higher
- **Width**: 800-1200 pixels (will be resized to 800px in README)
- **Filename**: `paper_preview.png`

After conversion, replace this file and commit the changes to GitHub.
