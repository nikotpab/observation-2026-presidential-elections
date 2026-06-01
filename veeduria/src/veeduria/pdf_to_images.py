from pathlib import Path
from PIL import Image
import pypdfium2 as pdfium


def pdf_to_images(pdf_path: Path, dpi: int = 200) -> list[Image.Image]:
    """Converts every page of a PDF to a PIL Image at the given DPI."""
    doc = pdfium.PdfDocument(str(pdf_path))
    scale = dpi / 72  # pdfium renders at 72 dpi by default
    images = []
    for page in doc:
        bitmap = page.render(scale=scale, rotation=0)
        images.append(bitmap.to_pil())
        page.close()
    doc.close()
    return images


def page_one(pdf_path: Path, dpi: int = 200) -> Image.Image:
    """Returns only page 1 — where the vote count table lives."""
    doc = pdfium.PdfDocument(str(pdf_path))
    scale = dpi / 72
    page = doc[0]
    bitmap = page.render(scale=scale, rotation=0)
    img = bitmap.to_pil()
    page.close()
    doc.close()
    return img
